#!/usr/bin/env python3
"""Inference-only SkillRise ScienceWorld reproduction.

This intentionally instantiates no actor, reference policy, optimizer, trainer,
or checkpoint writer.  It adapts a read-only vLLM engine to the one generation
method consumed by the official TrajectoryCollector.
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from agent_system.environments.skillrise_sciworld import make_envs
from agent_system.environments.skillrise_sciworld.group_loader import GroupLoader
from agent_system.multi_turn_rollout import TrajectoryCollector
from verl import DataProto


def load_env(env_path: Path) -> dict:
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def inference_worker(rank, conn, model_path, response_length, gpu_memory_utilization, max_model_len):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    from vllm import LLM, SamplingParams
    engine = LLM(
        model=model_path, tokenizer=model_path, tensor_parallel_size=1,
        dtype="bfloat16", seed=rank, gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, max_num_batched_tokens=16384, max_num_seqs=8,
        enable_chunked_prefill=True, enforce_eager=False, trust_remote_code=True)
    conn.send(("ready", rank))
    while True:
        message = conn.recv()
        if message is None:
            break
        request_index, raw_ids, validate = message
        params = SamplingParams(
            temperature=0.7 if validate else 1.0, top_p=1.0, top_k=-1,
            max_tokens=response_length, min_tokens=5 if validate else 0, n=1)
        output = engine.generate([{"prompt_token_ids": list(map(int, raw_ids))}], params, use_tqdm=False)[0]
        conn.send((request_index, list(output.outputs[0].token_ids)))
    conn.close()


class InferenceOnlyPolicy:
    def __init__(self, model_path, tokenizer, *, tensor_parallel_size, seed,
                 data_parallel_size, response_length, gpu_memory_utilization,
                 max_model_len, curate_via_api=False, api_env=None):
        self.tokenizer = tokenizer
        self.response_length = response_length
        self.curate_via_api = curate_via_api
        self.curate_client = None
        self.curate_model = "deepseek-chat"
        self._curate_failures = 0  # consecutive failed curate calls (E1)
        if curate_via_api:
            from openai import OpenAI
            api_env = api_env or {}
            self.curate_client = OpenAI(
                api_key=api_env.get("DEEPSEEK_API_KEY", ""),
                base_url=api_env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
            )
            self.curate_model = api_env.get("DEEPSEEK_MODEL", "deepseek-chat")
        if tensor_parallel_size != 1:
            raise ValueError("explicit inference pool requires tensor_parallel_size=1")
        ctx = mp.get_context("spawn")
        self.workers = []
        for rank in range(data_parallel_size):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=inference_worker,
                args=(rank, child, model_path, response_length,
                      gpu_memory_utilization, max_model_len))
            proc.start()
            self.workers.append((proc, parent))
        for proc, conn in self.workers:
            status, _ = conn.recv()
            if status != "ready":
                raise RuntimeError(f"inference worker {proc.pid} failed to initialize")

    # ------------------------------------------------------------------ #
    # DeepSeek curate (B1): route skill-distillation rows to the API.
    # ------------------------------------------------------------------ #
    def _is_curate_prompt(self, text: str) -> bool:
        # Marker unique to the SkillRise curate prompt
        # (SCIWORLD_CURATE_PROMPT); the SOLVE prompt only contains
        # "## Current Skill Document", which this does not match.
        return "maintaining a SKILL DOCUMENT" in text

    def _deepseek_curate(self, prompt_text: str) -> str:
        if self.curate_client is None:
            return ""
        user_msg = prompt_text + (
            "\n\nOutput ONLY the revised skill document inside "
            "<skill>...</skill> tags. Do not output anything else."
        )
        last_err = None
        # E1: escalate backoff inside a call; abort the run if the API is
        # down across several consecutive curate points (avoid producing a
        # silently-degraded run like the night of 2026-08-29/30).
        for attempt in range(5):
            try:
                resp = self.curate_client.chat.completions.create(
                    model=self.curate_model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert skill curator for "
                                    "ScienceWorld agents. Follow the user's "
                                    "instruction exactly."},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=self.response_length,
                )
                self._curate_failures = 0
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        self._curate_failures += 1
        if self._curate_failures >= 5:
            raise SystemExit(
                "[deepseek curate] DeepSeek API unreachable across "
                f"{self._curate_failures} consecutive curate calls "
                f"(last error: {last_err}) — aborting run to avoid corrupted "
                "data. Re-run when the API is back (experiment is resumable).")
        print(f"[deepseek curate] error after 5 retries: {last_err}", file=sys.stderr)
        return ""

    @torch.inference_mode()
    def generate_sequences_agent(self, prompts):
        active = prompts.non_tensor_batch["active_masks"]
        raw_ids = prompts.non_tensor_batch["raw_prompt_ids"]
        validate = bool(prompts.meta_info.get("validate", False))
        active_indices = [i for i, enabled in enumerate(active) if enabled]

        # One generation call is always a single phase (play or curate) for
        # every active row, so probing one row is enough to route the step.
        is_curate_step = False
        if active_indices and self.curate_via_api:
            probe = self.tokenizer.decode(raw_ids[active_indices[0]], skip_special_tokens=True)
            is_curate_step = self._is_curate_prompt(probe)

        generated = {}
        if is_curate_step:
            def _one(i):
                prompt_text = self.tokenizer.decode(raw_ids[i], skip_special_tokens=True)
                return i, self.tokenizer.encode(
                    self._deepseek_curate(prompt_text), add_special_tokens=False)
            with ThreadPoolExecutor(max_workers=min(4, len(active_indices))) as ex:
                for i, token_ids in ex.map(_one, active_indices):
                    generated[i] = token_ids
        else:
            for slot, i in enumerate(active_indices):
                self.workers[slot][1].send((i, raw_ids[i], validate))
            for slot, _ in enumerate(active_indices):
                i, token_ids = self.workers[slot][1].recv()
                generated[i] = token_ids

        rows = [generated.get(i, []) for i in range(len(active))]
        pad = self.tokenizer.pad_token_id
        responses = torch.full((len(rows), self.response_length), pad, dtype=torch.long)
        for i, row in enumerate(rows):
            row = row[:self.response_length]
            if row:
                responses[i, :len(row)] = torch.tensor(row, dtype=torch.long)

        prompt_ids = prompts.batch["input_ids"].cpu()
        prompt_attention = prompts.batch["attention_mask"].cpu()
        prompt_position = prompts.batch["position_ids"].cpu()
        response_attention = responses.ne(pad).to(prompt_attention.dtype)
        sequences = torch.cat((prompt_ids, responses), dim=-1)
        attention = torch.cat((prompt_attention, response_attention), dim=-1)
        delta = torch.arange(1, self.response_length + 1).unsqueeze(0)
        last_position = prompt_position[:, -1:].cpu()
        positions = torch.cat((prompt_position, last_position + delta), dim=-1)
        batch = TensorDict({
            "responses": responses,
            "input_ids": sequences,
            "attention_mask": attention,
            "position_ids": positions,
        }, batch_size=[len(rows)])
        return DataProto(batch=batch)

    def close(self):
        for _, conn in self.workers:
            conn.send(None)
        for proc, conn in self.workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
            conn.close()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initial_batch(size, tokenizer, *, validate):
    token = tokenizer.bos_token_id or tokenizer.eos_token_id
    return DataProto.from_single_dict({
        "input_ids": torch.full((size, 1), token, dtype=torch.long),
        "attention_mask": torch.ones((size, 1), dtype=torch.long),
        "position_ids": torch.zeros((size, 1), dtype=torch.long),
        "raw_prompt": np.array([[{"role": "user", "content": ""}]
                                for _ in range(size)], dtype=object),
        "data_source": np.array(["skillrise_sciworld"] * size, dtype=object),
    }, meta_info={"do_sample": True, "validate": validate,
                  "eos_token_id": tokenizer.eos_token_id,
                  "pad_token_id": tokenizer.pad_token_id})


def write_manifest(output_dir, group, started, finished, train_logs, val_logs, success):
    raw_train = output_dir / "rollouts" / "traj_logs" / "train" / "rollout_000000.jsonl"
    raw_val = output_dir / "rollouts" / "traj_logs" / "val" / "rollout_000001.jsonl"
    trials = []
    for i, log in enumerate(train_logs):
        phases = [{"phase": s["phase"], "task_pos": s["task_pos"], "step": s["step"]}
                  for s in log["steps"]]
        trials.append({
            "trial": i,
            "group_id": group["group_id"],
            "tasks": group["tasks"],
            "skill_updates": log["skills"],
            "phase_index": phases,
            "raw_jsonl": str(raw_train),
            "raw_jsonl_line": i + 1,
        })
    manifest = {
        "schema": "skillrise-pure-rollout-manifest-v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "group": group,
        "seed": 0,
        "K": 3,
        "N": 8,
        "started_at_epoch": started,
        "finished_at_epoch": finished,
        "wall_time_seconds": finished - started,
        "raw_exports": {"rollout": str(raw_train), "eval": str(raw_val)},
        "trials": trials,
        "eval": {"num_test_tasks": len(val_logs),
                 "metrics": {k: np.asarray(v).tolist() for k, v in success.items()}},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--group-file", default="data/groups/skillrise_sciworld_K3.jsonl")
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--data-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    ap.add_argument("--curate-via-api", action="store_true",
                    help="distill skill documents via DeepSeek API (B1) "
                         "instead of the Qwen vLLM engine")
    ap.add_argument("--seed-file", default=None,
                    help="D1: JSON file with per-row initial skills "
                         "{\"train\": [skill, ...], \"val\": [skill, ...]} "
                         "built by runtime/build_round2_seed.py; seeds env "
                         "rows with round-1 skills (+lessons) instead of ''")
    args = ap.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(0)

    api_env = {}
    if args.curate_via_api:
        api_env = load_env(Path(__file__).resolve().parent / ".env")
        if not api_env.get("DEEPSEEK_API_KEY"):
            raise SystemExit("--curate-via-api requires DEEPSEEK_API_KEY in runtime/.env")
        # E1: pre-flight check before the ~10 min engine init.
        import api_health
        if not api_health.check_deepseek(api_env):
            raise SystemExit(
                "[d1] DeepSeek API unreachable at start — aborting before "
                "engine init to avoid a corrupted run. Re-run when API is back.")

    cfg = OmegaConf.create({
        "data": {"train_batch_size": 1, "val_batch_size": 8,
                 "max_prompt_length": 10240, "max_response_length": 1024,
                 "truncation": "error", "return_raw_chat": True},
        "model": {"enable_thinking": False},
        "algorithm": {"step_gamma": 0.95, "traj_gamma": 0.6},
        "env": {"env_name": "skillrise_sciworld", "seed": 0,
                "max_steps": 30, "max_turns": 30, "num_attempts": 3,
                "rollout": {"n": 8}, "simplifications_preset": "easy",
                "resources_per_worker": {"num_cpus": 0.1},
                "max_env_per_rollout": 8, "meta_mode": "skillrise",
                "group_file": str(Path(args.group_file).resolve())},
        "trainer": {"rollout_data_dir": str(output_dir / "rollouts")},
        "inference": {"model_path": str(Path(args.model).resolve()),
                      "tensor_parallel_size": args.tensor_parallel_size,
                      "data_parallel_size": args.data_parallel_size,
                      "gpu_memory_utilization": args.gpu_memory_utilization,
                      "dtype": "bfloat16", "max_model_len": 11264,
                      "rollout_sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
                      "eval_sampling": {"temperature": 0.7, "top_p": 1.0, "top_k": -1}},
        "reproduction": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         "hostname": platform.node(), "gpus": list(range(8)),
                         "no_training": True},
    })
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    group = GroupLoader(cfg.env.group_file, 1, seed=0).next_batch()[0]
    (output_dir / "selected_group.json").write_text(json.dumps(group, indent=2, ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    except Exception:
        processor = None
    collector = TrajectoryCollector(cfg, tokenizer, processor)
    policy = InferenceOnlyPolicy(args.model, tokenizer,
        tensor_parallel_size=args.tensor_parallel_size, seed=0,
        data_parallel_size=args.data_parallel_size,
        response_length=cfg.data.max_response_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=cfg.inference.max_model_len,
        curate_via_api=args.curate_via_api, api_env=api_env)
    envs, val_envs = make_envs(cfg)
    if args.seed_file:
        seed = json.loads(Path(args.seed_file).read_text())
        envs.initial_skills = seed.get("train", [])
        val_envs.initial_skills = seed.get("val", [])
        print(f"[d1] seeded train rows={len(envs.initial_skills)} "
              f"val rows={len(val_envs.initial_skills)} from {args.seed_file}")
    started = time.time()
    try:
        _, train_logs = collector.multi_turn_loop(initial_batch(1, tokenizer, validate=False), policy, envs, is_train=False)
        # Preserve the official export split/name while identifying this first call as rollout.
        val_path = output_dir / "rollouts" / "traj_logs" / "val" / "rollout_000000.jsonl"
        train_path = output_dir / "rollouts" / "traj_logs" / "train" / "rollout_000000.jsonl"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.replace(train_path)
        _, val_logs = collector.multi_turn_loop(initial_batch(8, tokenizer, validate=True), policy, val_envs, is_train=False)
        # Reconstruct cumulative official pass@k directly from the exported per-attempt rewards.
        rewards = [[float(x.strip()) for x in log["reward"].strip("[]").split(",")] for log in val_logs]
        success = {f"pass@{k}": [any(r[:k]) for r in rewards] for k in (1, 2, 3)}
        finished = time.time()
        write_manifest(output_dir, group, started, finished, train_logs, val_logs, success)
    finally:
        envs.close()
        val_envs.close()
        policy.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
