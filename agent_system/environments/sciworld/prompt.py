# ScienceWorld prompts. PLAY follows the BEACON sciworld wording, but WITHOUT the
# `<think>` tag requirement (Qwen3 emits free-form/`<reasoning>` text; only
# `<action>` is validated, matching alfworld/webshop). REFLECT/CURATE follow the
# webshop reflection style (reflection inside <remark>, skill inside <skill>).

SCIWORLD_PLAY_PROMPT = """
You are an expert autonomous agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your task is to: {task_description}.{past_trajectories_reflections}{current_trajectory}
Your current observation is: {current_observation}.

Your admissible actions of the current situation are:
[
{admissible_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the task goal.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

SCIWORLD_REFLECT_PROMPT = """
You are an expert autonomous agent operating in the ScienceWorld environment, a text-based virtual environment centered on elementary-science tasks.
Your task is to: {task_description}.

You will be given the history of a past experience.
Your job is to **reflect on the past sequence**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** starting from the original initial state.

Below are the actions you took and the corresponding observations:
{current_trajectory}
The attempt reached a score of {score}/100 and the task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
"""


# ----- past trajectories / reflections injected into later attempts ----- #
PAST_EXPERIENCE_REFLECTION_TEMPLATE = """

On trial #{traj_idx}, the actions you took and the corresponding observations are:
{past_trajectory}
The attempt reached score {score}/100 and is NOT successfully completed. Your reflection is:
{reflection}"""

HISTORY_ONLY_TEMPLATE = """

On trial #{traj_idx}, the actions you took and the corresponding observations are:
{past_trajectory}
The attempt reached score {score}/100 and is NOT successfully completed."""

REFLECTION_ONLY_TEMPLATE = """

On trial #{traj_idx} (score {score}/100, not successfully completed). Your reflection is:
{reflection}"""


def parse_reflection(traj_idx, past_traj, reflection, scores, reflection_type='reflection_only'):
    if traj_idx == 0:
        return ''
    memories = []
    for _idx in range(traj_idx):
        # .get() so a missing reflection/history (e.g. GRPO val replays the SAME
        # task for pass@k WITHOUT a reflect phase) renders empty, not KeyError.
        _refl = reflection.get(_idx, '') if isinstance(reflection, dict) else ''
        _past = past_traj.get(_idx, '') if isinstance(past_traj, dict) else ''
        _score = int(scores.get(_idx, 0.0) * 100) if isinstance(scores, dict) else 0
        if reflection_type == 'history_and_reflection':
            memories.append(PAST_EXPERIENCE_REFLECTION_TEMPLATE.format(
                traj_idx=_idx + 1, past_trajectory=_past, score=_score, reflection=_refl))
        elif reflection_type == 'history_only':
            memories.append(HISTORY_ONLY_TEMPLATE.format(
                traj_idx=_idx + 1, past_trajectory=_past, score=_score))
        elif reflection_type == 'reflection_only':
            memories.append(REFLECTION_ONLY_TEMPLATE.format(
                traj_idx=_idx + 1, score=_score, reflection=_refl))
        else:
            raise NotImplementedError
    return ''.join(memories)


CURR_TRAJ_AT_TRAJ1 = """
Prior to this step, the actions you took and the corresponding observations are:
{current_trajectory}"""

CURR_TRAJ_AT_TRAJ2toN = """

Currently you're on trial #{traj_idx}, the actions you took and the corresponding observations are:
{current_trajectory}"""

TRAJ_2toN_INIT = """

Currently you're on trial #{traj_idx}, starting from the initial state."""


def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        return CURR_TRAJ_AT_TRAJ1.format(current_trajectory=curr_traj)
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        return CURR_TRAJ_AT_TRAJ2toN.format(traj_idx=traj_idx + 1, current_trajectory=curr_traj)


def get_sciworld_prompt(phase: str = 'play',
                        turn_idx: int = 0,
                        traj_idx: int = 0,
                        task_description: str = '',
                        current_observation: str = '',
                        action_history: str = '',
                        history_length: int = 0,
                        available_actions: str = '',
                        past_traj: dict = None,
                        reflection: dict = None,
                        scores: dict = None,
                        score: int = 0,
                        reflection_type: str = 'reflection_only'):
    assert phase in ['play', 'reflect']
    past_traj = past_traj or {}
    reflection = reflection or {}
    scores = scores or {}

    if phase == 'play':
        past_trajectories_reflections = parse_reflection(traj_idx, past_traj, reflection, scores, reflection_type)
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, action_history)
        return SCIWORLD_PLAY_PROMPT.format(
            task_description=task_description,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory,
            current_observation=current_observation,
            admissible_actions=available_actions,
        )
    else:
        return SCIWORLD_REFLECT_PROMPT.format(
            task_description=task_description,
            current_trajectory=action_history,
            score=int(score),
        )
