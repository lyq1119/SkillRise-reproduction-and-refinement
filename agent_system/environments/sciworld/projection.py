from typing import List


def sciworld_projection(actions: List[str], *args, phase: str = 'play', **kwargs):
    """Parse model output for ScienceWorld.

    Expected format: free-form step-by-step reasoning, then the action inside
    <action> </action> tags (aligned with alfworld/webshop: only <action> is
    required/validated; the reasoning is NOT wrapped in any tag).

    Returns (processed_actions, valids). valids[i]=1 iff a well-formed
    <action>...</action> block is present.

    Extra positional/keyword args (e.g. admissible_commands, phase) are accepted
    for signature-compatibility; SciWorld ignores them. For phase='reflect' the
    raw text is returned (the manager stores it).
    """
    valids = [0] * len(actions)
    processed_actions = []

    for i in range(len(actions)):
        original_str = actions[i]

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = original_str.find(start_tag)
        end_idx = original_str.find(end_tag)

        try:
            if start_idx == -1 or end_idx == -1:
                processed_actions.append(original_str[-20:])
                continue
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()
            processed_actions.append(extracted_action)
            valids[i] = 1
        except Exception:
            processed_actions.append(original_str[-20:])

    return processed_actions, valids
