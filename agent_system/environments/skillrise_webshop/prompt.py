"""Prompts for SkillRise cross-task meta-RL on WebShop.

Two roles share one policy:
  - SOLVE: read the current skill document + observation, reason, emit <action>
           (search[...] / click[...]).
  - CURATE: read the just-finished task trajectory + old skill document, rewrite an
            improved, task-agnostic skill document inside <skill>...</skill>.

All model-facing text is English. The skill document is a single markdown doc that
evolves in-context across the K tasks of one group (it is NOT persisted to disk during
training; each group/trial starts from an empty document).

The Curate prompt mirrors the ALFWorld version (abstract-not-memorize / ground-in-
trajectory / concise / suggested When-to-use·Workflow·Pitfalls structure, matching
SkillOS's curator prompt), and adds WebShop-specific distillation dimensions (search
query formulation, product selection, option configuration order, pre-purchase
constraint check).
"""

EMPTY_SKILL_PLACEHOLDER = "(The skill document is currently empty. No skills have been distilled yet.)"

# ---------------------------------------------------------------- #
# ----------------------------- SOLVE ---------------------------- #
# ---------------------------------------------------------------- #
WEBSHOP_SOLVE_PROMPT = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.

## Current Skill Document
The following skills were distilled from earlier shopping tasks of the SAME family. Use
them only when you are confident they apply to the current situation; ignore any skill
that does not fit.
{skill_document}

Your task is to: {task_description}.{current_trajectory}

Your admissible actions of the current situation are:
[
{admissible_actions}
].

Now it's your turn to take one action for the current step.
Your response should first be step-by-step reasoning about the current situation, then think carefully which admissible action best advances the shopping goal.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# ---------------------------------------------------------------- #
# ---------------------------- CURATE ---------------------------- #
# ---------------------------------------------------------------- #
WEBSHOP_CURATE_PROMPT = """
You are maintaining a SKILL DOCUMENT for an agent solving a family of related WebShop
online-shopping tasks. You have just observed one task attempt. Your job is to revise
the skill document so that it helps the agent solve the LATER tasks in this same family
(same product category) more reliably.

## Old Skill Document
{skill_document}

## Most Recent Task Attempt (the shopping goal, observations, and the actions the agent took)
{current_trajectory}
Outcome of this attempt: {outcome}

## Your job
Rewrite the skill document. Start from the old document above and refine it with what
this attempt revealed. Follow these rules strictly:

- ABSTRACT, do not memorize. Remove task-specific instances: replace concrete product
  names, brands, exact attribute values, and price thresholds (e.g. "rose red", "1.6
  ounce", "lower than 60.00 dollars") with general concepts ("the required color/size
  option", "the price ceiling").
- Ground every statement in what actually happened in the attempt above. Do not invent
  steps you did not observe.
- If the attempt failed, extract the failure point as a concrete pitfall.
- Be CONCISE. Do not pad with generic advice or vague tips. A short, precise document
  beats a long one. If this attempt revealed little new, keep the old document almost unchanged.
- Do NOT copy the trajectory verbatim. Distill reusable procedure, not history.

When useful, the document should capture WebShop-specific procedure:
- how to formulate a search query that encodes the goal's constraints,
- which search result to click (matching category/attributes, not just the first),
- the order to configure product options (color, size, count, etc.) before buying,
- verifying the product meets the required attributes AND the price ceiling BEFORE click[buy now],
- navigation patterns (when to go back, view options/description/features, paginate).

Suggested (not mandatory) structure -- use the headers that fit:

## When to use
<high-level situations this skill family applies to>
## Workflow
<ordered high-level steps that generalize across product instances>
## Pitfalls
<failure modes and when to deviate from the default workflow>

Write your reasoning first, then output the FULL revised skill document inside
<skill> </skill> tags.
"""


# ---------------------------------------------------------------- #
# --------------------- trajectory formatting -------------------- #
# ---------------------------------------------------------------- #
CURR_TRAJ_TEMPLATE = '''

Below are the last few actions and corresponding observations you have:
{current_trajectory}'''


def _format_skill(skill: str) -> str:
    skill = (skill or "").strip()
    return skill if skill else EMPTY_SKILL_PLACEHOLDER


def get_skillrise_webshop_prompt(phase: str = 'play',
                             turn_idx: int = 0,
                             task_description: str = '',
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
        prompt = WEBSHOP_SOLVE_PROMPT.format(
            skill_document=skill_document,
            task_description=task_description,
            current_trajectory=current_trajectory,
            admissible_actions=admissible_actions,
        )
    else:
        outcome = "SUCCESS" if won else "FAILURE (task not completed)"
        prompt = WEBSHOP_CURATE_PROMPT.format(
            skill_document=skill_document,
            current_trajectory=curr_traj,
            outcome=outcome,
        )

    return prompt.strip()
