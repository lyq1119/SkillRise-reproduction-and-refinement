from typing import List, Tuple, Dict
import os
import json
import torch
import numpy as np
from functools import partial

from .prompt import get_sciworld_prompt
from .envs import build_sciworld_envs
from .projection import sciworld_projection
from .memory import SimpleMemory
from ..base import EnvironmentManagerBase


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


class SciWorldEnvironmentManager(EnvironmentManagerBase):
    """LaMer-compatible ScienceWorld manager (GRPO + multi-attempt reflection).

    num_attempts == 1 -> pure GRPO play. num_attempts > 1 with do_reflection ->
    play / reflect / restart-same-task / play ... (LaMer). Success/Score metrics
    are produced by the inherited base.EnvironmentManagerBase.success_evaluator,
    which reads info['won'] and info['task_score'] set by the worker.
    """

    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts
        self.do_reflection = do_reflection
        self.init_states = [None for _ in range(self.num_processes)]
        # one memory buffer per attempt
        self.memories = [SimpleMemory() for _ in range(self.num_attempts)]
        self.reflections = [{} for _ in range(self.num_processes)]
        # scores[i][traj_idx] = best task_score (0..1) seen in that attempt
        self.scores = [{} for _ in range(self.num_processes)]
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        assert self.reflection_type in ['history_and_reflection', 'reflection_only', 'history_only']
        self.history_length = config.env.get('history_length', 15)
        # reflect sees a longer slice than play (aligned with skillrise curate=30).
        self.reflect_history_length = config.env.get('reflect_history_length', 30)
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0
        self.max_turns = config.env.get('max_turns', 30)
        self.tasks = ['' for _ in range(self.num_processes)]
        super().__init__(envs, projection_f, config)

    # ------------------------------------------------------------------ #
    def reset(self):
        text_obs, infos = self.envs.reset()
        self.tasks = [info.get('task_description', '') for info in infos]
        for info in infos:
            info['task_type'] = info.get('task_num', 0)

        for memory in self.memories:
            memory.reset(self.num_processes)
        self.reflections = [{} for _ in range(self.num_processes)]
        self.scores = [{} for _ in range(self.num_processes)]
        self.init_states = text_obs
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0

        full = self.build_text_obs(phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def restart(self):
        """Next attempt: same task replayed."""
        text_obs, infos = self.envs.restart()
        self.curr_traj_idx += 1
        self.curr_turn_idx = 0
        self.tasks = [info.get('task_description', '') for info in infos]
        for info in infos:
            info['task_type'] = info.get('task_num', 0)
        self.init_states = text_obs
        full = self.build_text_obs(phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def reflect(self):
        infos = [{"action_is_valid": True, "won": False,
                  "task_type": 0} for _ in range(self.num_processes)]
        observations = {
            'text': self.build_text_obs(phase='reflect'),
            'image': None,
            'anchor': ['reflection' for _ in range(self.num_processes)],
        }
        return observations, infos

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'reflect']

        if phase == 'reflect':
            reflections, valids = self.projection_f(text_actions, phase='reflect')
            for i, refl in enumerate(text_actions):
                # store the raw reflection text (trimmed) for the next attempt
                self.reflections[i][self.curr_traj_idx] = refl[:2000]
            infos = [{"action_is_valid": False, "won": False,
                      "is_action_valid": to_numpy(valids[i]), "task_type": 0}
                     for i in range(self.num_processes)]
            next_obs = {'text': '', 'image': None, 'anchor': ''}
            rewards = np.array(valids, dtype=np.float32)
            dones = np.array([False] * self.num_processes)
            return next_obs, rewards, dones, infos

        # phase == 'play'
        actions, valids = self.projection_f(text_actions, phase='play')
        text_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['task_type'] = info.get('task_num', info.get('task_type', 0))
            # track best score seen this attempt (for reflection display)
            s = float(info.get('task_score', 0.0))
            self.scores[i][self.curr_traj_idx] = max(
                self.scores[i].get(self.curr_traj_idx, 0.0), s)

        self.memories[self.curr_traj_idx].store({
            'text_obs': text_obs,
            'action': actions,
            'reward': rewards,
            'dones': dones,
            'won': [info['won'] for info in infos],
        })
        self.curr_turn_idx += 1

        full = self.build_text_obs(phase='play')
        next_obs = {'text': full, 'image': None, 'anchor': text_obs}
        return next_obs, to_numpy(rewards), to_numpy(dones), infos

    # ------------------------------------------------------------------ #
    def build_text_obs(self, phase: str = 'play') -> List[str]:
        assert phase in ['play', 'reflect']

        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
            curr_lens = [0 for _ in range(self.num_processes)]
        else:
            curr_trajs, curr_lens = self.memories[self.curr_traj_idx].fetch(self.history_length)

        # past attempts' trajectories (for history_* reflection types)
        past_trajs = [{} for _ in range(self.num_processes)]
        for traj_idx in range(self.curr_traj_idx):
            trajs, _ = self.memories[traj_idx].fetch(self.history_length)
            for i in range(self.num_processes):
                past_trajs[i][traj_idx] = trajs[i]

        # reflect distills the just-finished attempt over a longer window than play.
        if phase == 'reflect':
            reflect_trajs, _ = self.memories[self.curr_traj_idx].fetch(self.reflect_history_length)

        out = []
        for i in range(self.num_processes):
            if phase == 'reflect':
                obs = get_sciworld_prompt(
                    phase='reflect',
                    task_description=self.tasks[i],
                    action_history=reflect_trajs[i],
                    score=int(self.scores[i].get(self.curr_traj_idx, 0.0) * 100),
                )
            else:
                obs = get_sciworld_prompt(
                    phase='play',
                    turn_idx=self.curr_turn_idx,
                    traj_idx=self.curr_traj_idx,
                    task_description=self.tasks[i],
                    current_observation=self.init_states[i],
                    action_history=curr_trajs[i],
                    history_length=curr_lens[i],
                    available_actions=self.envs.get_admissible_commands[i],
                    past_traj=past_trajs[i],
                    reflection=self.reflections[i],
                    scores=self.scores[i],
                    reflection_type=self.reflection_type,
                )
            out.append(obs)
        return out


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    # variations index lives under RL-ExpSkill/data/sciworld (sibling of the LaMer
    # repo root), matching the ALFWORLD_DATA/group_file path convention. Resolve in
    # priority: config.env.sciworld_variations_idx -> $SCIWORLD_DATA -> default abs path.
    default_data = os.path.join(os.path.expanduser('~'), 'data', 'sciworld')
    sciworld_data = os.environ.get('SCIWORLD_DATA', default_data)
    variations_idx_path = config.env.get(
        'sciworld_variations_idx',
        os.path.join(sciworld_data, 'variations_idx', 'L0_idx.json'),
    )
    with open(variations_idx_path, 'r') as f:
        variations_idx = json.load(f)

    env_kwargs = {
        'jar_path': None,
        'env_step_limit': config.env.get('max_steps', 30),
        'simplifications_preset': config.env.get('simplifications_preset', 'easy'),
        'variations_idx': variations_idx,
    }

    rpw = config.env.get('resources_per_worker', None)
    if rpw is not None:
        resources_per_worker = {'num_cpus': rpw.get('num_cpus', 0.15)}
        if rpw.get('num_gpus', 0):
            resources_per_worker['num_gpus'] = rpw.get('num_gpus')
    else:
        resources_per_worker = {'num_cpus': 0.15}

    projection_f = partial(sciworld_projection)

    num_attempts = config.env.get('num_attempts', 1)
    do_reflection = config.env.get('do_reflection', num_attempts > 1)
    val_num_attempts = config.env.get('val_num_attempts', num_attempts)
    val_do_reflection = config.env.get('val_do_reflection', do_reflection)

    # Cap the live env-worker pool. Each worker is a JVM (ScienceWorld), so
    # train_batch*group_n + val_batch JVMs at once exhausts node memory/threads
    # ("pthread_create EAGAIN"). compute_groups_per_chunk shrinks the train pool;
    # the rollout loop then replays the batch in chunks reusing this pool.
    # max_env_per_rollout=0 (default) keeps the old all-at-once behavior.
    from agent_system.multi_turn_rollout.utils import compute_groups_per_chunk
    max_env = config.env.get('max_env_per_rollout', 0)
    groups_per_chunk = compute_groups_per_chunk(config.data.train_batch_size, group_n, max_env)

    train_envs = build_sciworld_envs(
        seed=config.env.seed, env_num=groups_per_chunk,
        group_n=group_n, resources_per_worker=resources_per_worker,
        is_train=True, env_kwargs=env_kwargs,
    )
    val_envs = build_sciworld_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
        group_n=1, resources_per_worker=resources_per_worker,
        is_train=False, env_kwargs=env_kwargs,
    )

    envs = SciWorldEnvironmentManager(train_envs, projection_f, num_attempts, do_reflection, config)
    envs.groups_per_chunk = groups_per_chunk
    val_envs_mgr = SciWorldEnvironmentManager(val_envs, projection_f, val_num_attempts, val_do_reflection, config)
    val_envs_mgr.groups_per_chunk = config.data.val_batch_size
    return envs, val_envs_mgr
