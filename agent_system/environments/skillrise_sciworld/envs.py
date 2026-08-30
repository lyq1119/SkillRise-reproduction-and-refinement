# Ray-based vectorised ScienceWorld env for SkillRise cross-task meta-RL.
# Superset of agent_system/environments/sciworld/envs.py: keeps the pool-sampling
# path (reset/restart, used by val repeat mode) AND adds explicit task loading so
# a train trial can play an ORDERED sequence of K different [task_id, variation]:
#   worker.load_game(task_id, variation)  — point this worker at a specific task
#   vec.load_games(pairs)                 — point every worker at its pos-th task
# (mirrors skillrise_alfworld/envs.py load_game/load_games, which use gamefile paths).

import ray
import gym
import numpy as np


class SciWorldWorker:
    """Ray remote actor hosting one ScienceWorldEnv instance."""

    def __init__(self, seed, env_kwargs):
        import sys
        import os
        import random

        # Spark ships an old py4j (0.10.9) on sys.path that shadows scienceworld's.
        sys.path = [p for p in sys.path if 'py4j-0.10.9-src' not in p]

        # JAVA_TOOL_OPTIONS is honored (it limits JVM threads, avoiding
        # pthread_create exhaustion). It must NOT contain -XX:CICompilerCount=1
        # without -XX:-TieredCompilation: that combo is illegal, the JVM aborts
        # before printing the gateway port, and py4j reads an empty line
        # ("invalid literal for int() ... b''"). env.sh sets a valid value.

        # py4j launches a bare `java`; ensure a Java-11 bin is on PATH/JAVA_HOME.
        jdk_path = (os.environ.get('SCIWORLD_JAVA_HOME')
                    or os.environ.get('JAVA_HOME')
                    or '')
        if os.path.isdir(os.path.join(jdk_path, 'bin')):
            os.environ['JAVA_HOME'] = jdk_path
            bindir = os.path.join(jdk_path, 'bin')
            if bindir not in os.environ.get('PATH', '').split(os.pathsep):
                os.environ['PATH'] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"

        from scienceworld import ScienceWorldEnv

        jar_path = env_kwargs.get('jar_path')
        env_step_limit = env_kwargs.get('env_step_limit', 100)
        simplifications_preset = env_kwargs.get('simplifications_preset', 'easy')

        self.env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
        self.taskNames = self.env.get_task_names()

        variations_idx = env_kwargs.get('variations_idx', [])
        if isinstance(variations_idx, dict):
            self.variations_idx = variations_idx.get('train', []) + variations_idx.get('test', [])
        else:
            self.variations_idx = variations_idx

        self.simplifications_preset = simplifications_preset
        self.task_num = 0

        random.seed(seed)
        self.rng = random.Random(seed)

    def _build_available_actions(self):
        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        return (
            f"Valid_actions: {valid_actions}, "
            f"OBJ needs to be replaced with one of the following objects: {valid_objs}\n"
            f"example: <action>focus on door</action>"
        )

    def _stamp_reset_info(self, obs, info, task_id):
        info = dict(info or {})
        self.task_num = task_id
        info['task_description'] = self.env.get_task_description()
        info['available_actions'] = self._build_available_actions()
        info['observation_text'] = obs
        info['possible_actions'] = self.env.get_valid_action_object_combinations()
        info['won'] = False
        info['task_num'] = task_id
        score = int(info.get('score', 0))
        info['score'] = score
        info['task_score'] = score / 100.0
        info['task_type'] = task_id
        return info

    def step(self, action):
        obs, _native_reward, done, info = self.env.step(action)
        info = dict(info or {})

        info['available_actions'] = self._build_available_actions()
        info['observation_text'] = obs
        info['possible_actions'] = self.env.get_valid_action_object_combinations()

        score = int(info.get('score', 0))
        info['score'] = score
        info['task_score'] = score / 100.0          # continuous completion, recorded for metrics
        # won only on episode termination with a full score (depend on done so the
        # reward is guaranteed terminal-only, regardless of any mid-episode score).
        info['won'] = bool(done and score >= 100)
        info['task_type'] = self.task_num

        # Strict binary terminal reward, aligned with alfworld/webshop (won -> 10, else 0).
        reward = 10.0 * float(info['won'])
        return obs, reward, done, info

    def reset(self, variation_idx):
        """Pool-sampling reset (val repeat mode): index into the split list."""
        if variation_idx is None or variation_idx >= len(self.variations_idx):
            task_id, task_variation = self.rng.choice(self.variations_idx)
        else:
            task_id, task_variation = self.variations_idx[variation_idx]
        return self.load_game(int(task_id), int(task_variation))

    def load_game(self, task_id, variation):
        """Load a SPECIFIC [task_id, variation] and reset into it (cross-task)."""
        taskName = self.taskNames[task_id]
        simplification_str = self.simplifications_preset if self.simplifications_preset else ""
        self.env.load(taskName, variation, simplification_str)
        obs, info = self.env.reset()
        return obs, self._stamp_reset_info(obs, info, task_id)

    def close(self):
        self.env.close()


class SkillRiseSciWorldEnvs(gym.Env):
    """Vectorised ScienceWorld env supporting BOTH explicit per-worker task
    loading (train cross mode via load_games) and pool sampling (val repeat mode
    via reset/restart)."""

    def __init__(self, seed, env_num, group_n, resources_per_worker,
                 is_train=True, env_kwargs=None):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}

        variations_idx = self._env_kwargs.get('variations_idx', {})
        if isinstance(variations_idx, dict):
            variations_list = (variations_idx.get('test', []) if not self.is_train
                               else variations_idx.get('train', []))
        else:
            variations_list = variations_idx
        self._env_kwargs['variations_idx'] = variations_list
        self.variation_idxs = range(len(variations_list))

        print(f"[SkillRiseSciWorld] Loaded {len(variations_list)} variations for "
              f"{'training' if self.is_train else 'testing'}")

        env_worker = ray.remote(**resources_per_worker)(SciWorldWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        self._last_idx = None
        self.prev_admissible_commands = ['' for _ in range(self.num_processes)]
        # EX: explicit pinned held-out task list for val (overrides pool sampling).
        self._val_pairs = None
        if not self.is_train:
            vp = env_kwargs.get('val_pairs') if env_kwargs else None
            if vp is not None:
                assert len(vp) == self.num_processes, \
                    f"#val_pairs({len(vp)}) != num_processes({self.num_processes})"
                self._val_pairs = [(int(p[0]), int(p[1])) for p in vp]

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    # ---- train cross mode: explicit ordered loading --------------------- #
    def load_games(self, pairs):
        """Point every worker at its assigned [task_id, variation]."""
        assert len(pairs) == self.num_processes, \
            f"#pairs({len(pairs)}) != num_processes({self.num_processes})"
        futures = [w.load_game.remote(int(p[0]), int(p[1]))
                   for w, p in zip(self._workers, pairs)]
        return self._collect_reset(futures)

    # ---- val repeat mode: pool sample + same-task retry ----------------- #
    def reset(self):
        if self._val_pairs is not None:
            # EX: explicit pinned held-out task list.
            self._last_idx = self._val_pairs
            return self.load_games(self._val_pairs)
        idx = self._rng.choice(self.variation_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
        self._last_idx = idx
        futures = [w.reset.remote(i) for w, i in zip(self._workers, idx)]
        return self._collect_reset(futures)

    def restart(self):
        assert self._last_idx is not None, "restart() called before reset()"
        if self._val_pairs is not None:
            futures = [w.load_game.remote(int(p[0]), int(p[1]))
                       for w, p in zip(self._workers, self._last_idx)]
            return self._collect_reset(futures)
        futures = [w.reset.remote(i) for w, i in zip(self._workers, self._last_idx)]
        return self._collect_reset(futures)

    def _collect_reset(self, futures):
        results = ray.get(futures)
        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            info_list.append(info)
            self.prev_admissible_commands[i] = info.get('available_actions', '')
        # 3-tuple (text, image=None, info) to match the manager's unpacking.
        return obs_list, None, info_list

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(f'Expected {self.num_processes} actions, got {len(actions)}')
        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, (obs, reward, done, info) in enumerate(results):
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_admissible_commands[i] = info.get('available_actions', '')
        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, '_closed', False):
            return
        close_futures = [w.close.remote() for w in self._workers]
        ray.get(close_futures)
        for w in self._workers:
            ray.kill(w)
        self._closed = True

    def __del__(self):
        self.close()


def build_skillrise_sciworld_envs(seed, env_num, group_n, resources_per_worker,
                              is_train=True, env_kwargs=None):
    return SkillRiseSciWorldEnvs(
        seed=seed, env_num=env_num, group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train, env_kwargs=env_kwargs,
    )
