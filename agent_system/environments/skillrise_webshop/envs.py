import ray
import gym
import numpy as np

# Mirrors agent_system/environments/webshop/envs.py. The ONLY cross-task addition is
# `load_sessions` (vec env): point every worker at an explicit session_idx instead of
# the random session draw that reset() gives, so one trial can play an ordered K-task
# group (the WebShop analogue of ALFWorld's load_games). Per-worker Ray resources are
# configurable to avoid CPU over-subscription.


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class SkillRiseWebshopWorker:
    """Ray remote actor that hosts one *WebAgentTextEnv* instance.

    Same as webshop/envs.py:WebshopWorker; `reset(session=idx)` already lets us point
    a worker at a SPECIFIC session (the cross-task meta-RL primitive), so no extra
    worker method is needed. CPU/GPU resources are assigned via .options() at
    construction (see SkillRiseWebshopEnvs).
    """

    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                     '..', 'webshop', 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        env_kwargs = dict(env_kwargs)
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)

    def step(self, action):
        """Execute a step in the environment."""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Rule-based reward: win -> 10, otherwise 0 (matches webshop/envs.py).
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0
        return obs, reward, done, info

    def reset(self, idx):
        """Reset the environment at a SPECIFIC session index (cross-task load)."""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info

    def restart(self):
        """Reload the SAME session (same-task retry for val pass@k)."""
        obs, info = self.env.reset(session=self.env.session)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info

    def get_available_actions(self):
        return self.env.get_available_actions()

    def get_goals(self):
        return self.env.server.goals

    def close(self):
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class SkillRiseWebshopEnvs(gym.Env):
    """Vectorised, Ray-based wrapper around *WebAgentTextEnv* for cross-task meta-RL.

    Adds `load_sessions(session_idx_list)` (one explicit session per worker) so a trial
    can play an ordered K-task group. `reset()`/`restart()` keep the original random/
    same-session behaviour for VAL (repeat mode).
    """

    def __init__(self, seed=0, env_num=1, group_n=1, is_train=True,
                 env_kwargs=None, resources_per_worker=None):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {
            'observation_mode': 'text', 'num_products': None}

        # Configurable per-worker resources (avoids env actors eating every CPU and
        # starving the trainer's GPU worker group). Default small CPU share.
        if resources_per_worker is None:
            resources_per_worker = {'num_cpus': 0.25}
        worker_cls = ray.remote(SkillRiseWebshopWorker).options(**resources_per_worker)

        # ALL workers MUST share the same seed: each worker shuffles its goals list
        # with random.seed(seed) (web_agent_text_env.py), and reset(session=idx) indexes
        # into that shuffle. The group file's session_idx identities were built with
        # seed=0, so every cross-task worker must reproduce that exact same shuffle --
        # otherwise reset(session=idx) loads a different product per worker/group. (The
        # original webshop used per-group seeds because it draws random sessions at
        # runtime and does not need reproducible session_idx identities.)
        self._workers = []
        for i in range(self.num_processes):
            worker = worker_cls.remote(seed, self._env_kwargs)
            self._workers.append(worker)

        # Eval split mirrors webshop/envs.py: held-out = range(500), train = rest.
        goals = ray.get(self._workers[0].get_goals.remote())
        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
        print(self.goal_idxs)

    # ------------------------------------------------------------------ #
    # cross-task addition
    # ------------------------------------------------------------------ #
    def load_sessions(self, session_idxs):
        """Point every worker at its assigned session_idx (len == num_processes).

        The WebShop analogue of ALFWorld's load_games: cross-task meta-RL needs an
        ordered sequence of different sessions, not the random draw reset() gives.
        """
        assert len(session_idxs) == self.num_processes, \
            "The num of session_idxs must be equal to the num of processes"
        futures = [w.reset.remote(int(idx)) for w, idx in zip(self._workers, session_idxs)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    # ------------------------------------------------------------------ #
    # base API
    # ------------------------------------------------------------------ #
    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(f'Expected {self.num_processes} actions, got {len(actions)}')
        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """Random session draw per env_num (VAL repeat mode). group_n rows share a session."""
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
        futures = [w.reset.remote(i) for w, i in zip(self._workers, idx)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def restart(self):
        """Reload the SAME session in every worker (val same-task retry)."""
        futures = [w.restart.remote() for w in self._workers]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        if getattr(self, '_closed', False):
            return
        ray.get([w.close.remote() for w in self._workers])
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True

    def __del__(self):
        self.close()


def build_skillrise_webshop_envs(seed=0, env_num=1, group_n=1, is_train=True,
                             env_kwargs=None, resources_per_worker=None):
    return SkillRiseWebshopEnvs(seed=seed, env_num=env_num, group_n=group_n,
                            is_train=is_train, env_kwargs=env_kwargs,
                            resources_per_worker=resources_per_worker)
