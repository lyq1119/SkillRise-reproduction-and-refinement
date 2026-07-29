# Ray-based vectorised wrapper around ScienceWorldEnv.
# Migrated from BEACON (agent_system/environments/env_package/sciworld/envs.py),
# with three deliberate changes for LaMer:
#   1. worker step() uses PURE TERMINAL score reward (reward = final_score/100 on
#      done, else 0) instead of BEACON's dense +1/+10 shaping.
#   2. won = (score >= 100) strict full completion (ScienceWorld score is 0..100
#      in this version), and task_score = score/100 is exposed for the base
#      success_evaluator's continuous Score metric.
#   3. restart() replays the SAME variation indices (LaMer retries the same task),
#      and get_admissible_commands is exposed for the manager.

import ray
import gym
import numpy as np


class SciWorldWorker:
    """Ray remote actor hosting one ScienceWorldEnv instance."""

    def __init__(self, seed, env_kwargs):
        import sys
        import os
        import random

        # Spark ships an old py4j (0.10.9) on sys.path that shadows the one
        # scienceworld needs; strip it before importing scienceworld.
        sys.path = [p for p in sys.path if 'py4j-0.10.9-src' not in p]

        # JAVA_TOOL_OPTIONS is honored (it limits JVM threads, avoiding
        # pthread_create exhaustion). It must NOT contain -XX:CICompilerCount=1
        # without -XX:-TieredCompilation: that combo is illegal, the JVM aborts
        # before printing the gateway port, and py4j reads an empty line
        # ("invalid literal for int() ... b''"). env.sh sets a valid value.

        # ScienceWorld launches its JVM via py4j, which calls a bare `java` from
        # PATH. Ray workers can inherit a sanitised PATH without Java 11, so we
        # ALWAYS prepend a known Java-11 bin to PATH (and set JAVA_HOME). The dir
        # is resolved from $SCIWORLD_JAVA_HOME, else $JAVA_HOME, else a default.
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

    def step(self, action):
        obs, _native_reward, done, info = self.env.step(action)
        info = dict(info or {})  # copy so we can mutate safely

        info['available_actions'] = self._build_available_actions()
        info['observation_text'] = obs
        info['possible_actions'] = self.env.get_valid_action_object_combinations()

        # ScienceWorld score is an integer 0..100 (this version converts the
        # internal 0-1 score to 0-100). It can go negative on task failure.
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
        if variation_idx is None or variation_idx >= len(self.variations_idx):
            task_id, task_variation = self.rng.choice(self.variations_idx)
        else:
            task_id, task_variation = self.variations_idx[variation_idx]

        taskName = self.taskNames[task_id]
        simplification_str = self.simplifications_preset if self.simplifications_preset else ""
        self.env.load(taskName, task_variation, simplification_str)
        obs, info = self.env.reset()
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

        return obs, info

    def close(self):
        self.env.close()


class SciWorldMultiProcessEnv(gym.Env):
    """Vectorised, Ray-based wrapper around ScienceWorldEnv."""

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
        # Shallow-copy so resolving the train/test split below doesn't mutate the
        # caller's dict (make_envs passes the SAME env_kwargs to train and val;
        # without the copy the train build would overwrite 'variations_idx' and
        # val would silently evaluate on the train split).
        self._env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}

        variations_idx = self._env_kwargs.get('variations_idx', {})
        if isinstance(variations_idx, dict):
            if not self.is_train:
                variations_list = variations_idx.get('test', [])
            else:
                variations_list = variations_idx.get('train', [])
        else:
            variations_list = variations_idx
        self._env_kwargs['variations_idx'] = variations_list
        self.variation_idxs = range(len(variations_list))

        print(f"[SciWorld] Loaded {len(variations_list)} variations for "
              f"{'training' if self.is_train else 'testing'}")

        env_worker = ray.remote(**resources_per_worker)(SciWorldWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        self._last_idx = None
        self.prev_admissible_commands = ['' for _ in range(self.num_processes)]

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

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

    def reset(self):
        # Sample env_num distinct variations, then replicate across the group.
        idx = self._rng.choice(self.variation_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
        self._last_idx = idx
        return self._reset_to(idx)

    def restart(self):
        """Replay the SAME variations (LaMer retries the same task per attempt)."""
        assert self._last_idx is not None, "restart() called before reset()"
        return self._reset_to(self._last_idx)

    def _reset_to(self, idx):
        futures = [w.reset.remote(i) for w, i in zip(self._workers, idx)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            info_list.append(info)
            self.prev_admissible_commands[i] = info.get('available_actions', '')
        return obs_list, info_list

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


def build_sciworld_envs(seed, env_num, group_n, resources_per_worker,
                        is_train=True, env_kwargs=None):
    return SciWorldMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train, env_kwargs=env_kwargs,
    )
