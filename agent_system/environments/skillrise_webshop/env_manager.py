from typing import List, Dict
from collections import defaultdict
import os
import torch
import numpy as np
from functools import partial

from .prompt import get_skillrise_webshop_prompt
from .projection import skillrise_webshop_projection
from .envs import build_skillrise_webshop_envs
from .group_loader import GroupLoader
from ..webshop.memory import SimpleMemory
from ..base import EnvironmentManagerBase


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


class SkillRiseWebshopEnvironmentManager(EnvironmentManagerBase):
    """Cross-task meta-RL manager for WebShop.

    One "trial" plays a fixed sequence of K different tasks (one group):
        Solve(x0) -> Curate(0) -> Solve(x1) -> Curate(1) -> ... -> Solve(x_{K-1})
    A single skill document S evolves in-context across the K tasks (one per row).
    N parallel trials replay the same group (consecutive N rows = same group) so
    advantages can be normalized group-relative across the N trials.

    Task identity is `session_idx` (loaded via env.reset(session=idx)); the WebShop
    analogue of ALFWorld's gamefile. `num_attempts` is reused by the rollout loop as
    the number of phases == K.
    """

    meta_mode = 'skillrise'

    def __init__(self, envs, projection_f, num_attempts, config, group_loader: GroupLoader = None,
                 task_mode='cross'):
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts            # == K
        self.K = num_attempts
        self.group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
        self.group_loader = group_loader
        # task_mode: 'cross' (train) plays K DIFFERENT sessions of a group, advancing the
        # session each position. 'repeat' (val) plays the SAME session K times, restarting
        # it each position (skill evolves via curate between attempts -> cumulative pass@1..K).
        self.task_mode = task_mode
        self.memories = [SimpleMemory() for _ in range(self.K)]
        self.skills = ['' for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_pos = 0
        self.max_turns = config.env.get('max_turns', 20)
        self.do_reflection = True  # rollout loop reads this; skillrise always inserts curate phases
        self.cur_admissible = [None for _ in range(self.num_processes)]
        super().__init__(envs, projection_f, config)

    # ------------------------------------------------------------------ #
    # group / session assignment
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

    def _sessions_at(self, pos: int) -> List[int]:
        return [self.row_group[row]['session_idxs'][pos] for row in range(self.num_processes)]

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
            text_obs, infos = self.envs.load_sessions(self._sessions_at(0))
        else:
            # val (repeat): draw one held-out session per row from the eval split
            text_obs, infos = self.envs.reset()
            self.row_task_type = ['all' for _ in range(self.num_processes)]

        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        self.cur_admissible = [info['available_actions'] for info in infos]
        self.tasks = self._extract_task(text_obs)
        anchor = self._format_obs(text_obs)
        full = self.build_text_obs(self.cur_admissible, phase='play')
        return {'text': full, 'image': None, 'anchor': anchor}, infos

    def advance(self):
        """train (cross): move to the next position, load each row's next session."""
        self.curr_pos += 1
        self.curr_turn_idx = 0
        text_obs, infos = self.envs.load_sessions(self._sessions_at(self.curr_pos))
        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        self.cur_admissible = [info['available_actions'] for info in infos]
        self.tasks = self._extract_task(text_obs)
        anchor = self._format_obs(text_obs)
        full = self.build_text_obs(self.cur_admissible, phase='play')
        return {'text': full, 'image': None, 'anchor': anchor}, infos

    def restart(self):
        """val (repeat): retry the SAME session; reload it in every worker."""
        self.curr_pos += 1
        self.curr_turn_idx = 0
        text_obs, infos = self.envs.restart()
        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        self.cur_admissible = [info['available_actions'] for info in infos]
        self.tasks = self._extract_task(text_obs)
        anchor = self._format_obs(text_obs)
        full = self.build_text_obs(self.cur_admissible, phase='play')
        return {'text': full, 'image': None, 'anchor': anchor}, infos

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

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'curate']
        if phase == 'play':
            actions, valids = self.projection_f(text_actions, phase='play')
            text_obs, rewards, dones, infos = self.envs.step(actions)
            text_obs_fmt = self._format_obs(text_obs)
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
                info['task_type'] = self.row_task_type[i]
            self.cur_admissible = [info['available_actions'] for info in infos]
            self.memories[self.curr_pos].store({
                'text_obs': text_obs_fmt,
                'action': actions,
                'reward': rewards,
                'dones': dones,
                'won': [info['won'] for info in infos],
            })
            self.curr_turn_idx += 1
            full = self.build_text_obs(self.cur_admissible, phase='play')
            anchor = text_obs_fmt
            next_obs = {'text': full, 'image': None, 'anchor': anchor}
            return next_obs, to_numpy(rewards), to_numpy(dones), infos
        else:
            skills, valids = self.projection_f(text_actions, phase='curate')
            for i, skill in enumerate(skills):
                if skill is not None:          # parse failure -> keep old skill, no penalty
                    self.skills[i] = skill
            infos = [{"won": False, "is_action_valid": to_numpy(valids[i]),
                      "task_type": self.row_task_type[i]}
                     for i in range(self.num_processes)]
            next_obs = {'text': '', 'image': None, 'anchor': ''}
            rewards = np.zeros(self.num_processes, dtype=np.float32)   # real reward set in credit assignment
            dones = np.array([False] * self.num_processes)
            return next_obs, rewards, dones, infos

    # ------------------------------------------------------------------ #
    # observation building
    # ------------------------------------------------------------------ #
    def _extract_task(self, text_obs: List[str]) -> List[str]:
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            try:
                assert parts[1] == 'Instruction:'
                tasks.append(parts[2])
            except Exception:
                tasks.append('')
        return tasks

    def _format_obs(self, text_obs: List[str]) -> List[str]:
        out = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            try:
                index = parts.index(self.tasks[i])
                reformatted = " [SEP] ".join(f"'{p}'" for p in parts[index + 1:])
            except Exception:
                reformatted = text_obs[i]
            out.append(reformatted)
        return out

    def _format_avail_actions(self, avail) -> str:
        actions = []
        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")
        if avail["has_search_bar"]:
            actions.append("search[<your query>]")
        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")
        return "\n".join(f"'{s}'," for s in actions)

    def build_text_obs(self, admissible_actions: List = None, phase='play') -> List[str]:
        assert phase in ['play', 'curate']
        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_pos].fetch(history_length=5, max_to_show=15)

        out = []
        for i in range(self.num_processes):
            if phase == 'play':
                adm = self._format_avail_actions(admissible_actions[i])
                obs = get_skillrise_webshop_prompt(
                    phase='play',
                    turn_idx=self.curr_turn_idx,
                    task_description=self.tasks[i],
                    curr_traj=curr_trajs[i],
                    admissible_actions=adm,
                    skill=self.skills[i],
                )
            else:
                traj, _ = self.memories[self.curr_pos].fetch(history_length=50, max_to_show=15)
                won_i = self._row_won(i, self.curr_pos)
                obs = get_skillrise_webshop_prompt(
                    phase='curate',
                    task_description=self.tasks[i],
                    curr_traj=traj[i],
                    skill=self.skills[i],
                    won=won_i,
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
                # val: same task retried K times -> cumulative pass@(pos+1) and the
                # best (max over attempts) raw 0~1 partial-match score so far.
                _won = False
                _score = 0.0
                for pos, won in enumerate(wons):
                    _won = _won or won
                    _score = max(_score, scores[pos])
                    success[f'success_rate[{pos}]'].append(_won)
                    success[f'{task_type}|success_rate[{pos}]'].append(_won)
                    success[f'task_score[{pos}]'].append(_score)
            else:
                for pos, won in enumerate(wons):
                    success[f'success_rate[{pos}]'].append(won)
                    success[f'{task_type}|success_rate[{pos}]'].append(won)
                    success[f'task_score[{pos}]'].append(scores[pos])

        return {k: np.array(v) for k, v in success.items()}


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    group_file = os.path.expandvars(config.env.get('group_file'))
    num_attempts = config.env.get('num_attempts')   # == K

    # Cap the train env pool so train_batch*group_n env actors don't exceed
    # max_env_per_rollout (the rollout loop splits the batch into chunks that each
    # reuse this pool). GroupLoader serves groups_per_chunk groups per reset(), so its
    # batch size must match the per-chunk env pool, not the full train batch.
    from agent_system.multi_turn_rollout.utils import compute_groups_per_chunk
    max_env = config.env.get('max_env_per_rollout', 0)
    groups_per_chunk = compute_groups_per_chunk(config.data.train_batch_size, group_n, max_env)

    train_loader = GroupLoader(group_file, groups_per_chunk, seed=config.env.seed)
    assert train_loader.K == num_attempts, (
        f"group K({train_loader.K}) != env.num_attempts({num_attempts})"
    )

    # Per-env-worker Ray resources (keep CPU small so env actors don't starve the trainer).
    rpw = config.env.get('resources_per_worker', None)
    if rpw is not None:
        resources_per_worker = {'num_cpus': rpw.get('num_cpus', 0.25)}
        if rpw.get('num_gpus', 0):
            resources_per_worker['num_gpus'] = rpw.get('num_gpus')
    else:
        resources_per_worker = {'num_cpus': 0.25}

    # WebShop product/instruction data (mirrors webshop/env_manager.py).
    ws = config.env.webshop
    base = os.path.join(os.path.dirname(__file__), '..', 'webshop', 'webshop', 'data')
    if ws.use_small:
        file_path = os.path.join(base, 'items_shuffle_1000.json')
        attr_path = os.path.join(base, 'items_ins_v2_1000.json')
    else:
        file_path = os.path.join(base, 'items_shuffle.json')
        attr_path = os.path.join(base, 'items_ins_v2.json')
    env_kwargs = {
        'observation_mode': 'text',
        'num_products': None,
        'human_goals': ws.human_goals,
        'file_path': file_path,
        'attr_path': attr_path,
    }

    projection_f = partial(skillrise_webshop_projection)

    # ---- TRAIN: SkillRise cross-task meta-RL (group of K DIFFERENT sessions + curate) ----
    _envs = build_skillrise_webshop_envs(seed=config.env.seed, env_num=groups_per_chunk,
                                     group_n=group_n, is_train=True, env_kwargs=env_kwargs,
                                     resources_per_worker=resources_per_worker)
    envs = SkillRiseWebshopEnvironmentManager(_envs, projection_f, num_attempts, config,
                                          group_loader=train_loader, task_mode='cross')
    envs.groups_per_chunk = groups_per_chunk

    # ---- VAL: SkillRise 'repeat' mode (held-out sessions, same task retried K times) ----
    _val_envs = build_skillrise_webshop_envs(seed=config.env.seed + 1000,
                                         env_num=config.data.val_batch_size, group_n=1,
                                         is_train=False, env_kwargs=env_kwargs,
                                         resources_per_worker=resources_per_worker)
    val_envs = SkillRiseWebshopEnvironmentManager(_val_envs, projection_f, num_attempts, config,
                                              group_loader=None, task_mode='repeat')
    val_envs.groups_per_chunk = config.data.val_batch_size

    import time
    time.sleep((groups_per_chunk * group_n + config.data.val_batch_size) * 0.1)
    return envs, val_envs
