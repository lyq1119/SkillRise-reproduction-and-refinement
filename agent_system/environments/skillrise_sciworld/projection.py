from typing import List


def skillrise_sciworld_projection(actions: List[str], action_pools: List[List[str]] = None, phase='play'):
    """Parse model outputs for SkillRise ScienceWorld.

    play  : extract the command inside <action>...</action> (only <action> is
            validated; reasoning is free-form, aligned with alfworld/webshop).
            NOT lowercased — ScienceWorld object names / numbers are case-sensitive.
    curate: extract the skill document inside <skill>...</skill>. On failure the
            skill is returned as None so the manager keeps the old document
            unchanged (no penalty).
    """
    assert phase in ['play', 'curate']
    if phase == 'play':
        valids = [0] * len(actions)
        out = []
        for i in range(len(actions)):
            s = actions[i]
            start_idx = s.find("<action>")
            end_idx = s.find("</action>")
            if start_idx == -1 or end_idx == -1:
                out.append(s[-30:])
                continue
            out.append(s[start_idx + len("<action>"):end_idx].strip())
            valids[i] = 1
        return out, valids
    else:
        valids = [0] * len(actions)
        skills = [None] * len(actions)
        for i in range(len(actions)):
            s = actions[i]
            start_idx = s.rfind("<skill>")
            end_idx = s.rfind("</skill>")
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                skills[i] = None  # parse failure: keep old skill, no penalty
            else:
                skills[i] = s[start_idx + len("<skill>"):end_idx].strip()[:4000]
                valids[i] = 1
        return skills, valids
