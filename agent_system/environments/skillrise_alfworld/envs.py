import os
import yaml
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
import ray

from ..alfworld.alfworld.agents.environment import get_environment

# Mirrors agent_system/environments/alfworld/envs.py. The ONLY cross-task additions are
# `load_game` (worker) / `load_games` (vec env): point a worker at a SPECIFIC gamefile
# instead of cycling the shuffled game pool, so one trial can play an ordered K-task group.
# Per-worker Ray resources are configurable (like SkillPilot) to avoid CPU over-subscription.

ALF_ACTION_LIST = ["pass", "goto", "pick", "put", "open", "close", "toggle",
                   "heat", "clean", "cool", "slice", "inventory", "examine", "look"]


def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i] *= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:, :, [2, 1, 0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors


def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward


@ray.remote
class SkillRiseAlfworldWorker:
    """Ray remote actor that holds one environment instance.

    Same as the LaMer AlfworldWorker, plus `load_game(gamefile)` which points this
    worker at an explicit gamefile (cross-task meta-RL needs an ordered sequence of
    different games, not the shuffled cycle that reset()/restart() give).
    Resources are assigned via .options() at construction (see SkillRiseAlfworldEnvs).
    """

    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)

    def step(self, action):
        """Execute a step in the environment"""
        actions = [action]
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos

    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos

    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()
        return image

    def restart(self):
        '''Get back to init state of the game'''
        self.env.last_commands = [None]
        self.env.obs, infos = self.env.batch_env.reset()
        obs = self.env.obs
        infos['observation_text'] = obs
        return obs, infos

    def load_game(self, gamefile):
        '''Load a SPECIFIC gamefile and reset into it (cross-task addition).'''
        self.env.batch_env.load([gamefile])
        self.env.last_commands = [None]
        self.env.obs, infos = self.env.batch_env.reset()
        obs = self.env.obs
        infos['observation_text'] = obs
        return obs, infos


class SkillRiseAlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, env_kwargs={},
                 resources_per_worker=None, is_train=True):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Configurable per-worker resources (avoids env actors eating every CPU and
        # starving the trainer's GPU worker group). Default small CPU share.
        if resources_per_worker is None:
            resources_per_worker = {'num_cpus': 0.25}
        worker_cls = SkillRiseAlfworldWorker.options(**resources_per_worker)
        self.workers = []
        for i in range(self.num_processes):
            worker = worker_cls.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def load_games(self, gamefiles):
        '''Point every worker at its assigned gamefile (len == num_processes).'''
        assert len(gamefiles) == self.num_processes, \
            "The num of gamefiles must be equal to the num of processes"

        futures = [w.load_game.remote(gf) for w, gf in zip(self.workers, gamefiles)]

        text_obs_list = []
        image_obs_list = []
        info_list = []
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def reset(self):
        '''Standard reset: each worker draws the next game from its shuffled pool.

        Used for VAL (repeat mode): pick a held-out task per row, same as the
        original LaMer/GiGPO evaluation. Train (cross mode) uses load_games.
        '''
        futures = [w.reset.remote() for w in self.workers]
        text_obs_list, info_list = [], []
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)
        image_obs_list = self.getobs() if self.multi_modal else None
        return text_obs_list, image_obs_list, info_list

    def restart(self):
        '''Reload the SAME game in every worker (same-task retry for val pass@k).'''
        futures = [w.restart.remote() for w in self.workers]
        text_obs_list, info_list = [], []
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)
        image_obs_list = self.getobs() if self.multi_modal else None
        return text_obs_list, image_obs_list, info_list

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        futures = [w.step.remote(actions[i]) for i, w in enumerate(self.workers)]

        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)
            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def getobs(self):
        futures = [w.getobs.remote() for w in self.workers]
        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        for worker in self.workers:
            ray.kill(worker)


def build_skillrise_alfworld_envs(alf_config_path, seed, env_num, group_n, env_kwargs={},
                              resources_per_worker=None, is_train=True):
    return SkillRiseAlfworldEnvs(alf_config_path, seed, env_num, group_n, env_kwargs,
                             resources_per_worker, is_train)
