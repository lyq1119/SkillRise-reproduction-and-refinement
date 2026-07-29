"""Prompts for SkillRise cross-task meta-RL on ScienceWorld.

Two roles share one policy:
  - SOLVE: read the current skill document + observation, reason, emit <action>.
  - CURATE: read the just-finished task trajectory + old skill document, rewrite
            an improved, task-agnostic skill document inside <skill>...</skill>.

The skill document evolves in-context across the K tasks of one group (not
persisted to disk; each group/trial starts empty).
"""

EMPTY_SKILL_PLACEHOLDER = "(The skill document is currently empty. No skills have been distilled yet.)"

# ---------------------------------------------------------------- #
# ----------------------------- SOLVE ---------------------------- #
# ---------------------------------------------------------------- #
SCIWORLD_SOLVE_PROMPT = """You are an expert autonomous agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.

## Current Skill Document
The following skills were distilled from earlier tasks of the SAME family. Use them only when you are confident they apply to the current situation; ignore any skill that does not fit.
{skill_document}

Your task is to: {task_description}.
Your current observation is: {current_observation}.{current_trajectory}
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the task goal.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

# ---------------------------------------------------------------- #
# ---------------------------- CURATE ---------------------------- #
# ---------------------------------------------------------------- #
SCIWORLD_CURATE_PROMPT = """You are maintaining a SKILL DOCUMENT for an agent solving a family of related ScienceWorld tasks. You have just observed one task attempt. Your job is to revise the skill document so that it helps the agent solve the LATER tasks in this same family more reliably.

## Old Skill Document
{skill_document}

## Most Recent Task Attempt (observations and the actions the agent took)
{current_trajectory}
Outcome of this attempt: {outcome}

## Your job
Rewrite the skill document. Start from the old document above and refine it with what this attempt revealed. Follow these rules strictly:

- ABSTRACT, do not memorize. Replace concrete instances (specific substance names, container colors, room names, numeric thresholds) with general concepts ("the target substance", "the designated container").
- Ground every statement in what actually happened in the attempt above. Do not invent steps you did not observe.
- If the attempt failed, extract the failure point as a concrete pitfall.
- Be CONCISE. A short, precise document beats a long one. If this attempt revealed little new, keep the old document almost unchanged.
- Do NOT copy the trajectory verbatim. Distill reusable procedure, not history.

Suggested (not mandatory) structure — use the headers that fit:

## When to use
<high-level situations this skill family applies to>
## Workflow
<ordered high-level steps that generalize across instances>
## Pitfalls
<failure modes and when to deviate from the default workflow>

Write your reasoning first, then output the FULL revised skill document inside <skill> </skill> tags."""


CURR_TRAJ_TEMPLATE = """
Prior to this step, the actions you took and the corresponding observations are:
{current_trajectory}"""


def _format_skill(skill: str) -> str:
    skill = (skill or "").strip()
    return skill if skill else EMPTY_SKILL_PLACEHOLDER


def get_skillrise_sciworld_prompt(phase: str = 'play',
                              turn_idx: int = 0,
                              task_description: str = '',
                              current_observation: str = '',
                              current_trajectory: str = '',
                              available_actions: str = '',
                              skill: str = '',
                              won: bool = False):
    assert phase in ['play', 'curate']
    skill_document = _format_skill(skill)

    if phase == 'play':
        traj = ("" if turn_idx == 0
                else CURR_TRAJ_TEMPLATE.format(current_trajectory=current_trajectory))
        return SCIWORLD_SOLVE_PROMPT.format(
            skill_document=skill_document,
            task_description=task_description,
            current_observation=current_observation,
            current_trajectory=traj,
            available_actions=available_actions,
        ).strip()
    else:
        outcome = "SUCCESS" if won else "FAILURE (task not fully completed)"
        return SCIWORLD_CURATE_PROMPT.format(
            skill_document=skill_document,
            current_trajectory=current_trajectory,
            outcome=outcome,
        ).strip()
