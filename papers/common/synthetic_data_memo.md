# Common Reading Memo: Best Practices on Synthetic Data (Liu et al., COLM 2024)

**Paper:** Best Practices and Lessons Learned on Synthetic Data for Language Models (Liu et al., COLM 2024)
**Role:** Operational reference for dataset-authoring decisions in Acts I–II

---

## Key Contributions

Liu et al. consolidate lessons from large-scale synthetic data programs across generation, filtering, and validation. Three findings are directly load-bearing for this project:

1. **Quality gates outperform volume.** Filtering synthetic data with an LLM judge consistently outperforms simply generating more data. The paper recommends a pointwise scoring gate (score each example independently on defined dimensions) rather than ranking-based selection.

2. **Diversity requires deliberate seeding.** Left to defaults, LLM generators produce clustered outputs around common patterns. Diversity must be engineered via varied prompts, multiple model families, or structured sampling across axes (domain, difficulty, format).

3. **Contamination is a generation-time problem, not a post-hoc one.** The paper recommends partitioning data before generation begins, so the generator never sees held-out scenarios. Retroactive decontamination is less reliable than structural prevention.

---

## Application to Tenacious-Bench

### Quality gate — `judge_filter.py`

The pointwise LLM judge in `generation/judge_filter.py` directly implements the paper's recommendation. Three dimensions (input_coherence, ground_truth_verifiability, rubric_clarity) are scored 1–5; tasks below 3.5 mean are dropped. All 200 tasks cleared this gate (mean 4.67). The paper validates this approach: filtering by quality score is more efficient than generating a larger raw corpus.

### Diversity seeding — four authoring modes

The paper's diversity warning maps directly to the rationale for four distinct authoring modes rather than one generator:
- `trace_derived` seeds from real failure traces (distribution anchor)
- `programmatic` varies prospect profiles via structured templates (axis coverage)
- `multi_llm_synthesis` uses DeepSeek V3 (different model family = different distributional bias)
- `adversarial` uses Claude Sonnet with explicit adversarial instructions (hard-case coverage)

Using a single generator (even with varied prompts) would have produced the clustering problem Liu et al. describe.

### Contamination — partition-before-generation

The contamination check in `generation/contamination_check.py` enforces the paper's structural prevention principle. Profile-level grouping ensures all task-type variants of the same prospect profile land in the same split before generation is complete. The 8-gram check (0 violations) and time-shift verification (93 public-signal tasks, all documented) satisfy the paper's held-out integrity requirements.

### One gap acknowledged

Liu et al. recommend human spot-checks on 5–10% of synthetic data before publishing. Our IRA protocol (`dataset/inter_rater_agreement.md`) covers 30 tasks (15% of dev split) for the two LLM-backed dimensions, which satisfies this recommendation for those dimensions. The five deterministic dimensions (κ = 1.0 by construction) are exempt.

---

*~420 words | Key finding applied: pointwise quality gate, diversity-by-seeding, partition-before-generation. Gap: human spot-check partially satisfied via IRA protocol.*
