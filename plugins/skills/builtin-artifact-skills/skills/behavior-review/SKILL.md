# Behavior Review

Use this skill for evidence-first second-pass review of behavior alarms, including AI复判, 人工复核, 二次审核, 二次研判, and alarm review follow-up tasks.

## Operating Rules

- Base the review decision on visual or structured evidence first. Do not confirm an incident from text rules alone.
- Treat detector labels, rule names, and original alarm text as candidates that need verification.
- If visual evidence is missing, unclear, blocked, or unrelated to the rule, return a need-more-evidence or manual-review conclusion.
- Suppress false positives when the target behavior is not visible, when the object is a lookalike, or when scene context contradicts the rule.
- Keep the output structured and concise.

## Required Reasoning

Follow this second-pass chain:

1. Text rule pre-check.
2. Evidence completeness check.
3. Model/rule consistency.
4. False positive suppression.
5. Final review conclusion.

## Output Contract

Return an object containing:

- `review_decision`: `confirm_incident`, `false_positive`, `need_more_evidence`, or `manual_review`.
- `risk_level`: `low`, `medium`, or `high`.
- `evidence_mode`: such as `visual_confirmed`, `text_only`, `insufficient_visual`, or `structured_only`.
- `evidence_gaps`: missing evidence or uncertainty points.
- `actions`: practical next steps.

