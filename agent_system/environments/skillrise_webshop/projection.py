from typing import List
import copy


def skillrise_webshop_projection(actions: List[str], action_pools: List[List[str]] = None, phase='play'):
    """Parse model outputs.

    play  : extract the command inside <action>...</action> (search[...]/click[...]).
    curate: extract the skill document inside <skill>...</skill>. On failure the skill
            is returned as None so the manager keeps the old document unchanged (no penalty).
    """
    assert phase in ['play', 'curate']
    if phase == 'play':
        valids = [0] * len(actions)
        actions = copy.deepcopy(actions)
        for i in range(len(actions)):
            actions[i] = actions[i].lower()
            start_tag, end_tag = "<action>", "</action>"
            start_idx = actions[i].find(start_tag)
            end_idx = actions[i].find(end_tag)
            if start_idx == -1 or end_idx == -1:
                actions[i] = actions[i][-20:]
                continue
            extracted = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            actions[i] = extracted
            valids[i] = 1
        return actions, valids
    else:
        valids = [0] * len(actions)
        skills = [None] * len(actions)
        for i in range(len(actions)):
            action = actions[i]
            start_tag, end_tag = "<skill>", "</skill>"
            start_idx = action.rfind(start_tag)
            end_idx = action.rfind(end_tag)
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                skills[i] = None  # parse failure: keep old skill, no penalty
            else:
                skills[i] = action[start_idx + len(start_tag):end_idx].strip()[:4000]
                valids[i] = 1
        return skills, valids
