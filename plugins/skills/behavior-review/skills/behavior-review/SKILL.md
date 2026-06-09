# Behavior Review

Use this skill for evidence-first second-pass review of behavior alarms, including AI复判, 人工复核, 二次审核, 二次研判, 连续复判, and alarm review follow-up tasks.

## Operating Rules

- Base the review decision on visual or structured evidence first. Do not confirm an incident from text rules alone.
- Treat detector labels, rule names, and original alarm text as candidates that need verification.
- If visual evidence is missing, unclear, too small, blocked, or unrelated to the rule, return a need-more-evidence or manual-review conclusion.
- Suppress false positives when the target behavior is not visible, when the object is a lookalike, or when the scene context contradicts the rule.
- Keep the output structured and concise. Prefer JSON-compatible values and avoid unrelated explanation.
- For continuous review, include the current round, next action, and whether polling should continue.

## Required Reasoning

Follow this second-pass chain:

1. Text rule pre-check: identify the original rule, detector label, and candidate behavior.
2. Evidence completeness check: state whether visual evidence exists and whether attachments are useful.
3. Model/rule consistency: compare visual evidence with the text rule and detector confidence.
4. False positive suppression: list visible contradiction signals or missing evidence.
5. Final review conclusion: choose confirm incident, false positive, need more evidence, or manual review.

## Output Contract

Return an object containing:

- `review_decision`: one of `confirm_incident`, `false_positive`, `need_more_evidence`, or `manual_review`.
- `risk_level`: `low`, `medium`, or `high`.
- `risk_score`: a numeric confidence/risk score from 0 to 1.
- `evidence_mode`: such as `visual_confirmed`, `text_only`, `insufficient_visual`, or `structured_only`.
- `second_review_logic`: ordered reasoning steps with short evidence notes.
- `evidence_gaps`: missing evidence or uncertainty points.
- `false_positive_signals`: visible reasons that weaken the alarm.
- `actions`: practical next steps.
- `continuous_review`: state for repeated polling when requested.

## Attachment Handling

- Use image or video attachments as primary evidence when present.
- If the runtime reports attachments but the model request has no image blocks, explicitly mark visual evidence as unavailable instead of pretending to inspect it.
- Mention only visible facts. Do not infer identity, age, intent, or health status beyond what is required by the rule.

