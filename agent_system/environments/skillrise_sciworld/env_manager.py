from typing import List, Dict
from collections import defaultdict
import os
import json
import torch
import numpy as np
from functools import partial

from .prompt import get_skillrise_sciworld_prompt
from .projection import skillrise_sciworld_projection
from .envs import build_skillrise_sciworld_envs
from .group_loader import GroupLoader
from .memory import SimpleMemory
from ..base import EnvironmentManagerBase


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


class SkillRiseSciWorldEnvironmentManager(EnvironmentManagerBase):
    """Cross-task meta-RL manager for ScienceWorld.

    One trial plays a fixed sequence of K different tasks (one group):
        Solve(x0) -> Curate(0) -> Solve(x1) -> ... -> Solve(x_{K-1})
    A single skill document S evolves in-context across the K tasks (one per row).
    N parallel trials replay the same group (consecutive N rows = same group) so
    advantages can be normalized group-relative across the N trials.

    `num_attempts` is reused by the rollout loop as the number of positions == K.
    """

    meta_mode = 'skillrise'

    def __init__(self, envs, projection_f, num_attempts, config, group_loader: GroupLoader = None,
                 task_mode='cross', group_n_override=None):
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts            # == K
        self.K = num_attempts
        # group_n: N parallel trials per group. Train replays each group N times
        # (config.env.rollout.n); val repeat uses 1 sequence per group.
        if group_n_override is not None:
            self.group_n = group_n_override
        else:
            self.group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
        self.group_loader = group_loader
        # task_mode: 'cross' (train) plays K DIFFERENT tasks of a group, advancing
        # the [task_id,variation] pair each position. 'repeat' (val) plays the SAME
        # task K times (skill evolves via curate between attempts).
        self.task_mode = task_mode
        # one memory buffer per task position
        self.memories = [SimpleMemory() for _ in range(self.K)]
        # evolving skill document, one per row
        self.skills = ['' for _ in range(self.num_processes)]
        self.curate_history_length = config.env.get('curate_history_length', 30)
        self.play_history_length = config.env.get('history_length', 15)
        self.curr_turn_idx = 0
        self.curr_pos = 0
        self.max_turns = config.env.get('max_turns', 30)
        self.do_reflection = True  # rollout loop reads this; skillrise always inserts curate
        self.tasks = ['' for _ in range(self.num_processes)]
        self.init_states = ['' for _ in range(self.num_processes)]
        super().__init__(envs, projection_f, config)

    # ------------------------------------------------------------------ #
    # group / pair assignment
    # ------------------------------------------------------------------ #
    def _assign_groups(self):
        groups = self.group_loader.next_batch()
        assert len(groups) * self.group_n == self.num_processes, (
            f"#groups({len(groups)}) * N({self.group_n}) != num_processes({self.num_processes})"
        )
        self.row_group = [None] * self.num_processes
        self.row_task_type = [None] * self.num_processes
        for gi, g in enumerate(groups):
            for r in range(self.group_n):
                row = gi * self.group_n + r
                self.row_group[row] = g
                self.row_task_type[row] = g['task_type']

    def _pairs_at(self, pos: int) -> List[List[int]]:
        return [self.row_group[row]['pairs'][pos] for row in range(self.num_processes)]

    # ------------------------------------------------------------------ #
    # rollout-loop entry points
    # ------------------------------------------------------------------ #
    def reset(self):
        for memory in self.memories:
            memory.reset(self.num_processes)
        self.skills = ['' for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_pos = 0

        if self.task_mode == 'cross':
            self._assign_groups()
            text_obs, _, infos = self.envs.load_games(self._pairs_at(0))
        else:
            # val repeat: pool-sample one held-out task per row from the test split.
            text_obs, _, infos = self.envs.reset()
            self.row_task_type = [info.get('task_num', 0) for info in infos]

        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        self.init_states = [info.get('observation_text', o) for info, o in zip(infos, text_obs)]
        self.tasks = [info.get('task_description', '') for info in infos]
        full = self.build_text_obs(phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def curate(self):
        """Between tasks: ask the policy to distill the just-finished trajectory."""
        infos = [{"won": False, "task_type": self.row_task_type[i]}
                 for i in range(self.num_processes)]
        observations = {
            'text': self.build_text_obs(phase='curate'),
            'image': None,
            'anchor': ['curate' for _ in range(self.num_processes)],
        }
        return observations, infos

    def advance(self):
        """train (cross): move to the next position, load each row's next pair."""
        self.curr_pos += 1
        self.curr_turn_idx = 0
        text_obs, _, infos = self.envs.load_games(self._pairs_at(self.curr_pos))
        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        self.init_states = [info.get('observation_text', o) for info, o in zip(infos, text_obs)]
        self.tasks = [info.get('task_description', '') for info in infos]
        full = self.build_text_obs(phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def restart(self):
        """val (repeat): retry the SAME task; reload it in every worker."""
        self.curr_pos += 1
        self.curr_turn_idx = 0
        text_obs, _, infos = self.envs.restart()
        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        self.init_states = [info.get('observation_text', o) for info, o in zip(infos, text_obs)]
        self.tasks = [info.get('task_description', '') for info in infos]
        full = self.build_text_obs(phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'curate']
        if phase == 'play':
            actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands, phase='play')
            text_obs, rewards, dones, infos = self.envs.step(actions)
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
                info['task_type'] = self.row_task_type[i]
            self.memories[self.curr_pos].store({
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
        else:  # phase == 'curate'
            skills, valids = self.projection_f(text_actions, phase='curate')
            for i, skill in enumerate(skills):
                if skill is not None:          # parse failure -> keep old skill, no penalty
                    self.skills[i] = skill
            infos = [{"won": False, "is_action_valid": to_numpy(valids[i]),
                      "task_type": self.row_task_type[i]}
                     for i in range(self.num_processes)]
            next_obs = {'text': '', 'image': None, 'anchor': ''}
            rewards = np.zeros(self.num_processes, dtype=np.float32)  # real reward set in credit assignment
            dones = np.array([False] * self.num_processes)
            return next_obs, rewards, dones, infos

    # ------------------------------------------------------------------ #
    # observation building
    # ------------------------------------------------------------------ #
    def build_text_obs(self, phase='play') -> List[str]:
        assert phase in ['play', 'curate']
        if phase == 'play':
            if self.curr_turn_idx == 0:
                curr_trajs = ['' for _ in range(self.num_processes)]
            else:
                curr_trajs, _ = self.memories[self.curr_pos].fetch(self.play_history_length)
        else:
            curr_trajs, _ = self.memories[self.curr_pos].fetch(self.curate_history_length)

        out = []
        for i in range(self.num_processes):
            if phase == 'play':
                obs = get_skillrise_sciworld_prompt(
                    phase='play', turn_idx=self.curr_turn_idx,
                    task_description=self.tasks[i],
                    current_observation=self.init_states[i],
                    current_trajectory=curr_trajs[i],
                    available_actions=self.envs.get_admissible_commands[i],
                    skill=self.skills[i],
                )
            else:  # curate
                won_i = self._row_won(i, self.curr_pos)
                obs = get_skillrise_sciworld_prompt(
                    phase='curate', current_trajectory=curr_trajs[i],
                    skill=self.skills[i], won=won_i,
                )
            out.append(obs)
        return out

    def _row_won(self, row: int, pos: int) -> bool:
        data = self.memories[pos]._data
        if data is None or row >= len(data):
            return False
        for rec in data[row]:
            if rec.get('won'):
                return True
        return False

    # ------------------------------------------------------------------ #
    # success metrics
    # ------------------------------------------------------------------ #
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        success = defaultdict(list)

        for bs in range(batch_size):
            task_type = total_infos[bs][0]['task_type']
            wons = [False for _ in range(self.K)]
            scores = [0.0 for _ in range(self.K)]
            for i in reversed(range(len(total_batch_list[bs]))):
                item = total_batch_list[bs][i]
                if item['active_masks'] and item['phase'] == 'play':
                    info = total_infos[bs][i]
                    pos = item['traj_idx']
                    wons[pos] = wons[pos] or info['won']
                    if info.get('task_score', 0.0):
                        scores[pos] = max(scores[pos], float(info['task_score']))

            if self.task_mode == 'repeat':
                # val: same task retried K times -> cumulative pass@(pos+1)
                _won, _score = False, 0.0
                for pos in range(self.K):
                    _won = _won or wons[pos]
                    _score = max(_score, scores[pos])
                    success[f'success_rate[{pos}]'].append(float(_won))
                    success[f'task_score[{pos}]'].append(_score)
            else:
                # cross: position pos is a DIFFERENT task -> per-position pass@1.
                padded = self._padded_flags(bs)
                for pos in range(self.K):
                    if padded is not None and pos < len(padded) and padded[pos]:
                        continue
                    success['success_rate'].append(float(wons[pos]))
                    success['task_score'].append(scores[pos])
                    success[f'success_rate[{pos}]'].append(float(wons[pos]))
                    success[f'task_score[{pos}]'].append(scores[pos])
                    success[f'{task_type}|success_rate[{pos}]'].append(float(wons[pos]))

        return {k: np.array(v) for k, v in success.items()}

    def _padded_flags(self, row: int):
        rg = getattr(self, 'row_group', None)
        if not rg or row >= len(rg) or rg[row] is None:
            return None
        tasks = rg[row].get('tasks', [])
        return [bool(t.get('padded', False)) for t in tasks]


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    # variations index (same source as the plain sciworld env).
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

    projection_f = partial(skillrise_sciworld_projection)
    num_attempts = config.env.get('num_attempts')   # == K

    # Cap the live env-worker pool. Each worker is a JVM (ScienceWorld), so
    # train_batch*group_n JVMs at once exhausts node memory/threads
    # ("pthread_create EAGAIN"). compute_groups_per_chunk shrinks the train pool
    # to `groups_per_chunk` groups; the rollout loop replays the batch in chunks
    # reusing this pool. The GroupLoader must serve groups_per_chunk groups/batch
    # so _assign_groups fills exactly the pool. max_env_per_rollout=0 -> no cap.
    from agent_system.multi_turn_rollout.utils import compute_groups_per_chunk
    max_env = config.env.get('max_env_per_rollout', 0)
    groups_per_chunk = compute_groups_per_chunk(config.data.train_batch_size, group_n, max_env)

    group_file = os.path.expandvars(config.env.get('group_file'))
    train_loader = GroupLoader(group_file, groups_per_chunk, seed=config.env.seed)
    assert train_loader.K == num_attempts, (
        f"group K({train_loader.K}) != env.num_attempts({num_attempts})"
    )

    # TRAIN: cross-task (group of K different tasks + curate)
    _envs = build_skillrise_sciworld_envs(
        seed=config.env.seed, env_num=groups_per_chunk, group_n=group_n,
        resources_per_worker=resources_per_worker, is_train=True, env_kwargs=env_kwargs,
    )
    envs = SkillRiseSciWorldEnvironmentManager(_envs, projection_f, num_attempts, config,
                                           group_loader=train_loader, task_mode='cross')
    envs.groups_per_chunk = groups_per_chunk

    # VAL: repeat mode (same held-out task retried K times from the test split).
    _val_envs = build_skillrise_sciworld_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1,
        resources_per_worker=resources_per_worker, is_train=False, env_kwargs=env_kwargs,
    )
    val_envs = SkillRiseSciWorldEnvironmentManager(_val_envs, projection_f, num_attempts, config,
                                               group_loader=None, task_mode='repeat',
                                               group_n_override=1)
    val_envs.groups_per_chunk = config.data.val_batch_size
    return envs, val_envs
