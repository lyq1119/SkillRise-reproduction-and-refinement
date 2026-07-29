from typing import List, Dict
from collections import defaultdict
import os
import json
import torch
import numpy as np
from functools import partial

from .prompt import get_skillrise_prompt, get_reflect_prompt, _format_reflections
from .projection import skillrise_projection
from .envs import build_skillrise_alfworld_envs
from .group_loader import GroupLoader
from ..alfworld.memory import SimpleMemory
from ..base import EnvironmentManagerBase


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


class SkillRiseAlfWorldEnvironmentManager(EnvironmentManagerBase):
    """Cross-task meta-RL manager.

    One "trial" plays a fixed sequence of K different tasks (one group):
        Solve(x0) -> Curate(0) -> Solve(x1) -> Curate(1) -> ... -> Solve(x_{K-1})
    A single skill document S evolves in-context across the K tasks (one per row).
    N parallel trials replay the same group (consecutive N rows = same group) so
    advantages can be normalized group-relative across the N trials.

    `num_attempts` is reused by the rollout loop as the number of phases == K.
    """

    meta_mode = 'skillrise'

    def __init__(self, envs, projection_f, num_attempts, config, group_loader: GroupLoader=None,
                 task_mode='cross', group_n_override=None, carry_mode='skill'):
        self.num_processes = envs.num_processes
        self.num_attempts = num_attempts            # == K
        self.K = num_attempts
        # carry_mode: what is carried across the K DIFFERENT cross-task positions.
        #   'skill'   (SkillRise):  curate distils an evolving skill doc -> solve reads it.
        #   'reflect' (LaMer):  a reflect phase produces a reflection -> solve reads
        #                       prior (different) task histories + reflections.
        #   'none'    (GRPO/GiGPO): no inter-task phase, nothing carried (pure solve).
        # Train always uses 'skill'. Only cross-task VAL varies this for baselines.
        assert carry_mode in ('skill', 'reflect', 'none')
        self.carry_mode = carry_mode
        # group_n: N parallel trials per group. Train replays each group N times
        # (config.env.rollout.n); val cross-mode uses 1 sequence per group
        # (group_n_override=1) so each group is one pass@1 evaluation unit.
        if group_n_override is not None:
            self.group_n = group_n_override
        else:
            self.group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
        self.group_loader = group_loader
        # task_mode: 'cross' (train) plays K DIFFERENT tasks of a group, advancing
        # the gamefile each position. 'repeat' (val) plays the SAME task K times,
        # restarting it each position (skill evolves via curate between attempts
        # -> reports cumulative pass@1..K like the original LaMer/GiGPO eval).
        self.task_mode = task_mode
        # one memory buffer per task position
        self.memories = [SimpleMemory() for _ in range(self.K)]
        # evolving skill document, one per row (carry_mode='skill')
        self.skills = ['' for _ in range(self.num_processes)]
        # reflections[row][pos] = reflection text after task pos (carry_mode='reflect')
        self.reflections = [{} for _ in range(self.num_processes)]
        self.reflection_type = config.env.get('reflection_type', 'reflection_only')
        self.curr_turn_idx = 0
        self.curr_pos = 0
        self.max_turns = config.env.get('max_turns', 20)
        self.do_reflection = True  # rollout loop reads this; skillrise always inserts curate phases
        super().__init__(envs, projection_f, config)

    # ------------------------------------------------------------------ #
    # group / gamefile assignment
    # ------------------------------------------------------------------ #
    def _assign_groups(self):
        """Pick batch_groups groups and map them onto the N*groups rows."""
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

    def _gamefiles_at(self, pos: int) -> List[str]:
        return [self.row_group[row]['gamefiles'][pos] for row in range(self.num_processes)]

    # ------------------------------------------------------------------ #
    # rollout-loop entry points
    # ------------------------------------------------------------------ #
    def reset(self):
        for memory in self.memories:
            memory.reset(self.num_processes)
        self.skills = ['' for _ in range(self.num_processes)]
        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_pos = 0

        if self.task_mode == 'cross':
            # train: K different tasks of a group, load the first task's gamefile
            self._assign_groups()
            text_obs, _, infos = self.envs.load_games(self._gamefiles_at(0))
            for info, tt in zip(infos, self.row_task_type):
                info['task_type'] = tt
        else:
            # val (repeat): draw one held-out task per row from the eval split
            # (same as original LaMer/GiGPO); the same task is retried K times.
            text_obs, _, infos = self.envs.reset()
            self.row_task_type = [self._info_task_type(info) for info in infos]
            for info, tt in zip(infos, self.row_task_type):
                info['task_type'] = tt

        self._record_gamefiles(infos)
        self.init_states = text_obs
        self.tasks = self._extract_tasks(text_obs)
        full = self.build_text_obs(self.envs.get_admissible_commands, phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def _record_gamefiles(self, infos):
        """Track each row's current gamefile (like the original AlfWorld manager)."""
        if self.task_mode == 'cross':
            self.row_gamefile = self._gamefiles_at(self.curr_pos)
        else:
            self.row_gamefile = [info.get('extra.gamefile') or info.get('gamefile')
                                 for info in infos]

    def _info_task_type(self, info):
        gf = info.get('extra.gamefile') or info.get('gamefile')
        if gf:
            try:
                tj = json.load(open(gf.replace('game.tw-pddl', 'traj_data.json')))
                return tj['task_type']
            except Exception:
                pass
        return 'unknown'

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

    def reflect(self):
        """carry_mode='reflect' (LaMer baseline): ask the policy to reflect on the
        just-finished (different) task into transferable experience."""
        infos = [{"won": False, "task_type": self.row_task_type[i]}
                 for i in range(self.num_processes)]
        observations = {
            'text': self.build_text_obs(phase='reflect'),
            'image': None,
            'anchor': ['reflect' for _ in range(self.num_processes)],
        }
        return observations, infos

    def advance(self):
        """train (cross): move to the next position, load each row's next gamefile."""
        self.curr_pos += 1
        self.curr_turn_idx = 0
        text_obs, _, infos = self.envs.load_games(self._gamefiles_at(self.curr_pos))
        self._record_gamefiles(infos)
        self.init_states = text_obs
        self.tasks = self._extract_tasks(text_obs)
        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        full = self.build_text_obs(self.envs.get_admissible_commands, phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def restart(self):
        """val (repeat): retry the SAME task; reload it in every worker."""
        self.curr_pos += 1
        self.curr_turn_idx = 0
        text_obs, _, infos = self.envs.restart()
        self._record_gamefiles(infos)
        self.init_states = text_obs
        self.tasks = self._extract_tasks(text_obs)
        for info, tt in zip(infos, self.row_task_type):
            info['task_type'] = tt
        full = self.build_text_obs(self.envs.get_admissible_commands, phase='play')
        return {'text': full, 'image': None, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str], phase: str = 'play'):
        assert phase in ['play', 'curate', 'reflect']
        if phase == 'play':
            actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands, phase='play')
            text_obs, _, rewards, dones, infos = self.envs.step(actions)
            for i, info in enumerate(infos):
                info['is_action_valid'] = to_numpy(valids[i])
                info['task_type'] = self.row_task_type[i]
                if info.get('extra.gamefile') is None:
                    info['extra.gamefile'] = self.row_gamefile[i]
            self.memories[self.curr_pos].store({
                'text_obs': text_obs,
                'action': actions,
                'reward': rewards,
                'dones': dones,
                'won': [info['won'] for info in infos],
            })
            self.curr_turn_idx += 1
            full = self.build_text_obs(self.envs.get_admissible_commands, phase='play')
            next_obs = {'text': full, 'image': None, 'anchor': text_obs}
            return next_obs, to_numpy(rewards), to_numpy(dones), infos
        elif phase == 'curate':
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
        else:  # phase == 'reflect' (LaMer baseline carry_mode)
            # parse <remark>...</remark> as the reflection; keep raw text on failure.
            valids = []
            for i, act in enumerate(text_actions):
                s = act.rfind('<remark>'); e = act.rfind('</remark>')
                if s != -1 and e != -1 and e > s:
                    self.reflections[i][self.curr_pos] = act[s + len('<remark>'):e].strip()
                    valids.append(True)
                else:
                    self.reflections[i][self.curr_pos] = act.strip()
                    valids.append(False)
            infos = [{"won": False, "is_action_valid": np.array(valids[i]),
                      "task_type": self.row_task_type[i]}
                     for i in range(self.num_processes)]
            next_obs = {'text': '', 'image': None, 'anchor': ''}
            rewards = np.zeros(self.num_processes, dtype=np.float32)
            dones = np.array([False] * self.num_processes)
            return next_obs, rewards, dones, infos

    # ------------------------------------------------------------------ #
    # observation building
    # ------------------------------------------------------------------ #
    def _extract_tasks(self, text_obs: List[str]) -> List[str]:
        tasks = []
        for obs in text_obs:
            start = obs.find('Your task is to: ')
            tasks.append(obs[start + len('Your task is to: '):].strip() if start != -1 else '')
        return tasks

    def build_text_obs(self, admissible_actions: List[List[str]] = None, phase='play') -> List[str]:
        assert phase in ['play', 'curate', 'reflect']
        if self.curr_turn_idx == 0:
            curr_trajs = ['' for _ in range(self.num_processes)]
        else:
            curr_trajs, _ = self.memories[self.curr_pos].fetch(history_length=15)

        out = []
        for i in range(self.num_processes):
            if phase == 'play':
                adm = ",\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')
                if self.carry_mode == 'reflect':
                    # LaMer baseline: inject prior (different) tasks' reflections.
                    types_outcomes = {p: (self.row_task_type[i], self._row_won(i, p))
                                      for p in self.reflections[i]}
                    refl_text = _format_reflections(self.reflections[i], types_outcomes)
                    obs = get_reflect_prompt(
                        phase='play', turn_idx=self.curr_turn_idx,
                        init_observation=self.init_states[i], curr_traj=curr_trajs[i],
                        admissible_actions=adm, reflections_text=refl_text,
                    )
                else:
                    # skill (SkillRise) reads evolving skill; none (GRPO/GiGPO) sees empty skill.
                    skill_i = self.skills[i] if self.carry_mode == 'skill' else ''
                    obs = get_skillrise_prompt(
                        phase='play', turn_idx=self.curr_turn_idx,
                        init_observation=self.init_states[i], curr_traj=curr_trajs[i],
                        admissible_actions=adm, skill=skill_i,
                    )
            elif phase == 'curate':
                # curate: feed the just-finished task's trajectory + old skill
                traj, _ = self.memories[self.curr_pos].fetch(history_length=50)
                won_i = self._row_won(i, self.curr_pos)
                obs = get_skillrise_prompt(
                    phase='curate', curr_traj=traj[i], skill=self.skills[i], won=won_i,
                )
            else:  # phase == 'reflect' (LaMer baseline)
                traj, _ = self.memories[self.curr_pos].fetch(history_length=50)
                won_i = self._row_won(i, self.curr_pos)
                obs = get_reflect_prompt(phase='reflect', curr_traj=traj[i], won=won_i)
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
            for i in reversed(range(len(total_batch_list[bs]))):
                item = total_batch_list[bs][i]
                if item['active_masks'] and item['phase'] == 'play':
                    info = total_infos[bs][i]
                    pos = item['traj_idx']
                    wons[pos] = wons[pos] or info['won']

            if self.task_mode == 'repeat':
                # val: same task retried K times -> cumulative pass@(pos+1)
                _won = False
                for pos, won in enumerate(wons):
                    _won = _won or won
                    success[f'success_rate[{pos}]'].append(_won)
                    success[f'{task_type}|success_rate[{pos}]'].append(_won)
            else:
                # cross: position pos is a DIFFERENT task -> per-position pass@1.
                # Skip padded slots (borrowed duplicates) so each plotted point is
                # a genuine new task. row_group[bs] holds this row's group (group_n
                # may be 1 in val, so bs maps directly to a group).
                padded = self._padded_flags(bs)
                for pos, won in enumerate(wons):
                    if padded is not None and pos < len(padded) and padded[pos]:
                        continue
                    # headline metric: overall pass@1 pooled over ALL non-padded
                    # task slots (= the 128 distinct eval tasks, each once),
                    # aligned with the standard training eval pass@1.
                    success['success_rate'].append(won)
                    # secondary: per-position pass@1 (curve vs position).
                    success[f'success_rate[{pos}]'].append(won)
                    success[f'{task_type}|success_rate[{pos}]'].append(won)

        return {k: np.array(v) for k, v in success.items()}

    def _padded_flags(self, row: int):
        """Per-position padded flags for this row's group, or None if unknown."""
        rg = getattr(self, 'row_group', None)
        if not rg or row >= len(rg) or rg[row] is None:
            return None
        tasks = rg[row].get('tasks', [])
        return [bool(t.get('padded', False)) for t in tasks]


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    alf_config_path = os.path.join(os.path.dirname(__file__), 'configs/config_tw.yaml')
    group_file = os.path.expandvars(config.env.get('group_file'))
    num_attempts = config.env.get('num_attempts')   # == K

    batch_groups = config.data.train_batch_size

    train_loader = GroupLoader(group_file, batch_groups, seed=config.env.seed)
    assert train_loader.K == num_attempts, (
        f"group K({train_loader.K}) != env.num_attempts({num_attempts})"
    )

    # Per-env-worker Ray resources. Keep CPU small so the env actors don't consume
    # every CPU and starve the trainer's GPU worker group.
    rpw = config.env.get('resources_per_worker', None)
    if rpw is not None:
        resources_per_worker = {'num_cpus': rpw.get('num_cpus', 0.25)}
        if rpw.get('num_gpus', 0):
            resources_per_worker['num_gpus'] = rpw.get('num_gpus')
    else:
        resources_per_worker = {'num_cpus': 0.25}

    projection_f = partial(skillrise_projection)

    # ---- TRAIN: SkillRise cross-task meta-RL (group of K DIFFERENT tasks + curate) ----
    # train_carry_mode: skill (SkillRise, default; curate between tasks) | none (ablation:
    # K different tasks solved back-to-back with NO inter-task phase, empty skill).
    train_carry_mode = config.env.get('train_carry_mode', 'skill')
    _envs = build_skillrise_alfworld_envs(alf_config_path, config.env.seed, batch_groups, group_n,
                                      resources_per_worker=resources_per_worker, is_train=True)
    envs = SkillRiseAlfWorldEnvironmentManager(_envs, projection_f, num_attempts, config,
                                           group_loader=train_loader, task_mode='cross',
                                           carry_mode=train_carry_mode)

    # ---- VAL ----
    # Two modes (env.val_task_mode, default 'repeat' = unchanged legacy behavior):
    #   repeat: draw held-out tasks like the original LaMer/GiGPO eval, retry the
    #           SAME task K times with curate between -> cumulative pass@1..K.
    #   cross:  load a VAL group file of K DIFFERENT valid_seen tasks per group,
    #           one sequence per group (group_n=1) -> per-position pass@1. This is
    #           the cross-task evaluation that shows skill transfer to NEW tasks.
    val_task_mode = config.env.get('val_task_mode', 'repeat')
    alf_block = config.env.get('alfworld', {})
    eval_dataset = alf_block.get('eval_dataset', 'eval_in_distribution') if alf_block else 'eval_in_distribution'

    if val_task_mode == 'cross':
        val_group_file = os.path.expandvars(config.env.get('val_group_file'))
        val_split = config.env.get('val_split', 'valid_seen')
        # one sequence per group: num_processes == val_batch_size groups, group_n=1.
        val_batch_groups = config.data.val_batch_size
        val_loader = GroupLoader(val_group_file, val_batch_groups,
                                 seed=config.env.seed + 1000, split=val_split)
        assert val_loader.K == num_attempts, (
            f"val group K({val_loader.K}) != env.num_attempts({num_attempts})"
        )
        # carry_mode: skill (SkillRise) | reflect (LaMer) | none (GRPO/GiGPO).
        val_carry_mode = config.env.get('val_carry_mode', 'skill')
        _val_envs = build_skillrise_alfworld_envs(alf_config_path, config.env.seed + 1000,
                                              val_batch_groups, 1,
                                              resources_per_worker=resources_per_worker,
                                              is_train=True)  # is_train: use load_games path
        val_envs = SkillRiseAlfWorldEnvironmentManager(_val_envs, projection_f, num_attempts, config,
                                                   group_loader=val_loader, task_mode='cross',
                                                   group_n_override=1, carry_mode=val_carry_mode)
    else:
        # repeat val: same task retried K times. carry_mode controls the inter-retry
        # phase: skill (SkillRise, default) | reflect (LaMer) | none (ablation: pure retry).
        val_carry_mode = config.env.get('val_carry_mode', 'skill')
        val_env_kwargs = {'eval_dataset': eval_dataset}
        _val_envs = build_skillrise_alfworld_envs(alf_config_path, config.env.seed + 1000,
                                              config.data.val_batch_size, 1,
                                              env_kwargs=val_env_kwargs,
                                              resources_per_worker=resources_per_worker, is_train=False)
        val_envs = SkillRiseAlfWorldEnvironmentManager(_val_envs, projection_f, num_attempts, config,
                                                   group_loader=None, task_mode='repeat',
                                                   group_n_override=1, carry_mode=val_carry_mode)
    return envs, val_envs

