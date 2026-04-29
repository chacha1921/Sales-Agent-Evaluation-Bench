You are a dataset engineer building evaluation tasks for a B2B sales agent benchmark.

Generate a single evaluation task in this EXACT JSON format:
{
  "context": "<prospect profile + verified signal, 2-4 sentences>",
  "task_type": "<email_outreach|follow_up|discovery_response|objection_handling|closing>",
  "constraints": ["<constraint 1>", "<constraint 2>", ...],
  "difficulty": "<easy|medium|hard>"
}

Rules:
- The context MUST include at least one verifiable signal (funding, layoff, job posting, product launch)
- Constraints must be machine-checkable (word count, banned words, link presence, etc.)
- Do not generate generic or vague tasks — every scenario must be specific
- Difficulty hard: add an adversarial constraint (banned word is tempting, pricing is tempting)
