# Path B Memo: Preference Leakage (Li et al., 2025)

**Paper:** Preference Leakage: A Contamination Problem in LLM-as-a-Judge (Li et al., 2025)
**Role:** Risk register for the generate-then-judge pipeline; what to avoid

---

## Key Contribution

Li et al. identify a specific contamination failure mode in LLM-as-a-judge pipelines: **preference leakage**. When the same model (or same model family) generates training data and later judges evaluation outputs, the judge inflates scores for outputs that stylistically resemble the generator's distribution — independently of actual quality. The effect size can be 0.3–0.8 points on a 5-point scale, large enough to produce false conclusions about method superiority.

Three leakage vectors are identified:

1. **Same-model leakage:** Generator and judge are the same model. Judge assigns high scores to its own output style.
2. **Same-family leakage:** Generator and judge are different models from the same family (e.g., GPT-4o generates, GPT-4-turbo judges). Shared RLHF priors inflate agreement.
3. **Prompt-template leakage:** Judge prompt template was present in the generator's training data, causing the judge to reproduce the expected scoring pattern from memory rather than evaluating the actual output.

---

## Application to Tenacious-Bench

### Current pipeline audit

| Stage | Model | Risk |
|---|---|---|
| Programmatic generation | `template_expansion` (no LLM) | No leakage — rule-based |
| Multi-LLM synthesis | DeepSeek V3 via OpenRouter | Low — DeepSeek family ≠ Claude |
| Adversarial generation | Claude Sonnet 4.6 | **Medium** — same family as Haiku judge |
| Judge filter | Claude Haiku 4.5 | Judges Sonnet-generated tasks |
| Tone/objection judge | Claude Haiku 4.5 | Judges all tasks |

The adversarial tasks (12/200) are the only within-family pair (Sonnet generates, Haiku judges). Li et al. find same-family leakage is smaller than same-model leakage but measurable.

### Mitigations already in place

1. **Model family separation for the majority of tasks (94%):** Programmatic + DeepSeek tasks are judged by Claude Haiku — cross-family, no leakage risk.
2. **`generation_model` field logged per task:** Every task records which model generated it. This enables stratified analysis: scores can be reported separately for `deepseek/deepseek-chat` tasks vs `claude-sonnet-4-6` tasks to detect family-specific inflation.
3. **IRA with human annotators:** The inter-rater agreement protocol provides a human ground truth that is independent of any model family. If Haiku is inflating scores for Sonnet-generated tasks, human annotators would not share that bias. The κ = 1.000 result provides some reassurance, though the 30-task IRA sample is not stratified by generation model.

### Remaining risk and recommended action

The 12 adversarial tasks (all Claude Sonnet) should be treated as potentially inflated in the final results. Recommended: report held-out ablation scores stratified by `generation_model`. If adversarial-task scores are systematically higher than DeepSeek-task scores at similar difficulty, preference leakage is the likely explanation.

---

*~420 words | Risk identified: 12/200 adversarial tasks are same-family (Sonnet→Haiku). Mitigations: model field logged, cross-family for 94% of tasks, human IRA independent of model. Action: stratify final results by generation_model.*
