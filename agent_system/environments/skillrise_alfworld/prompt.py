"""Prompts for SkillRise cross-task meta-RL on ALFWorld.

Two roles share one policy:
  - SOLVE: read the current skill document + observation, reason, emit <action>.
  - CURATE: read the just-finished task trajectory + old skill document, rewrite
            an improved, task-agnostic skill document inside <skill>...</skill>.

All model-facing text is English. The skill document is a single markdown doc
that evolves in-context across the K tasks of one group (it is NOT persisted to
disk during training; each group/trial starts from an empty document).
"""

EMPTY_SKILL_PLACEHOLDER = "(The skill document is currently empty. No skills have been distilled yet.)"

# ---------------------------------------------------------------- #
# ----------------------------- SOLVE ---------------------------- #
# ---------------------------------------------------------------- #
ALFWORLD_SOLVE_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.

## Current Skill Document
The following skills were distilled from earlier tasks of the SAME family. Use
them only when you are confident they apply to the current situation; ignore any
skill that does not fit.
{skill_document}

{init_observation}{current_trajectory}

Your admissible actions of the current situation are:
[{admissible_actions}]

Now it's your turn to take an action.

- Your response should first be step-by-step reasoning about the current situation.
- Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags.
"""

# ---------------------------------------------------------------- #
# ---------------------------- CURATE ---------------------------- #
# ---------------------------------------------------------------- #
ALFWORLD_CURATE_PROMPT = """
You are maintaining a SKILL DOCUMENT for an agent solving a family of related
ALFRED household tasks. You have just observed one task attempt. Your job is to
revise the skill document so that it helps the agent solve the LATER tasks in
this same family more reliably.

## Old Skill Document
{skill_document}

## Most Recent Task Attempt (observations and the actions the agent took)
{current_trajectory}
Outcome of this attempt: {outcome}

## Your job
Rewrite the skill document. Start from the old document above and refine it with
what this attempt revealed. Follow these rules strictly:

- ABSTRACT, do not memorize. Remove task-specific instances: replace concrete
  object names, receptacle ids, and location numbers (e.g. "soapbottle 1",
  "drawer 3") with general concepts ("the target object", "a receptacle").
- Ground every statement in what actually happened in the attempt above. Do not
  invent steps you did not observe.
- If the attempt failed, extract the failure point as a concrete pitfall.
- Be CONCISE. Do not pad with generic advice or vague tips. A short, precise
  document beats a long one. If this attempt revealed little new, keep the old
  document almost unchanged.
- Do NOT copy the trajectory verbatim. Distill reusable procedure, not history.

Suggested (not mandatory) structure — use the headers that fit:

## When to use
<high-level situations this skill family applies to>
## Workflow
<ordered high-level steps that generalize across instances>
## Pitfalls
<failure modes and when to deviate from the default workflow>

Write your reasoning first, then output the FULL revised skill document inside
<skill> </skill> tags.
"""


# ---------------------------------------------------------------- #
# --------------------- trajectory formatting -------------------- #
# ---------------------------------------------------------------- #
CURR_TRAJ_TEMPLATE = '''
Below are the actions you took and the corresponding observations:
{current_trajectory}'''


def _format_skill(skill: str) -> str:
    skill = (skill or "").strip()
    return skill if skill else EMPTY_SKILL_PLACEHOLDER


# ---------------------------------------------------------------- #
# ------- LaMer-style REFLECT baseline (cross-task variant) ------- #
# ---------------------------------------------------------------- #
# Used only by carry_mode='reflect' val (the LaMer baseline run through the
# skillrise cross-task loader). Unlike the original LaMer reflect (same task
# retried), here the prior tasks were DIFFERENT but same-family, so the wording
# frames the reflection as transferable experience, not "retry this task".
REFLECT_SOLVE_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.

## Experience From Earlier Tasks
Earlier you attempted other tasks from the SAME family (different objects /
receptacles / scenes). Below is what you learned. Apply it when it helps; ignore
anything that does not fit the current task.
{reflections}

{init_observation}{current_trajectory}

Your admissible actions of the current situation are:
[{admissible_actions}]

Now it's your turn to take an action.

- Your response should first be step-by-step reasoning about the current situation.
- Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags.
"""

REFLECT_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.

You have just finished a task from a family of related household tasks. Reflect
on this attempt and extract TRANSFERABLE experience that will help you solve the
LATER, DIFFERENT tasks in this same family (different objects/receptacles/scenes,
same overall procedure).

## Most Recent Task Attempt (observations and the actions you took)
{current_trajectory}
Outcome of this attempt: {outcome}

- First reason step-by-step about what worked, what went wrong, and which parts
  generalize across tasks of this family (abstract away the specific object/receptacle).
- Then write concise transferable guidance for the next, different task.
- End with your reflection inside <remark> </remark> tags.
"""

REFLECT_ONE_TEMPLATE = '''
- After an earlier task ({task_type}, outcome: {outcome}), you noted:
{reflection}'''


def _format_reflections(reflections_dict, types_outcomes):
    """reflections_dict: {pos: text}; types_outcomes: {pos: (task_type, won)}."""
    if not reflections_dict:
        return "\n(No earlier experience yet.)"
    parts = []
    for pos in sorted(reflections_dict):
        tt, won = types_outcomes.get(pos, ('a task', False))
        parts.append(REFLECT_ONE_TEMPLATE.format(
            task_type=tt,
            outcome="SUCCESS" if won else "FAILURE",
            reflection=(reflections_dict[pos] or "").strip(),
        ))
    return "".join(parts)


def get_reflect_prompt(phase='play', turn_idx=0, init_observation='', curr_traj='',
                       admissible_actions='', reflections_text='', won=False):
    """LaMer reflect-baseline prompt (cross-task). phase in {'play','reflect'}."""
    if phase == 'play':
        current_trajectory = ("" if turn_idx == 0
                              else CURR_TRAJ_TEMPLATE.format(current_trajectory=curr_traj))
        return REFLECT_SOLVE_PROMPT.format(
            reflections=reflections_text or "\n(No earlier experience yet.)",
            init_observation=init_observation,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        ).strip()
    else:
        outcome = "SUCCESS" if won else "FAILURE (task not completed)"
        return REFLECT_PROMPT.format(current_trajectory=curr_traj, outcome=outcome).strip()


def get_skillrise_prompt(phase: str = 'play',
                     turn_idx: int = 0,
                     init_observation: str = '',
                     curr_traj: str = '',
                     admissible_actions: str = '',
                     skill: str = '',
                     won: bool = False):
    assert phase in ['play', 'curate']
    skill_document = _format_skill(skill)

    if phase == 'play':
        if turn_idx == 0:
            current_trajectory = ""
        else:
            current_trajectory = CURR_TRAJ_TEMPLATE.format(current_trajectory=curr_traj)
        prompt = ALFWORLD_SOLVE_PROMPT.format(
            skill_document=skill_document,
            init_observation=init_observation,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        )
    else:
        outcome = "SUCCESS" if won else "FAILURE (task not completed)"
        prompt = ALFWORLD_CURATE_PROMPT.format(
            skill_document=skill_document,
            current_trajectory=curr_traj,
            outcome=outcome,
        )

    return prompt.strip()
