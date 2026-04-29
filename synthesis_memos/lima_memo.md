# Synthesis Memo: LIMA — Less Is More for Alignment (Zhou et al., 2023)

**Paper:** LIMA: Less Is More for Alignment (NeurIPS 2023)
**Design choice critiqued:** Diversity as the primary curation criterion for fine-tuning data (§4, "Data Collection and Curation")

---

## The Design Decision

The central claim of LIMA is the "Superficial Alignment Hypothesis": that alignment is primarily a matter of learning the format and style of responses, and that the base model already contains the knowledge needed to answer most queries (§2, "The Superficial Alignment Hypothesis"). The practical conclusion: fine-tuning on 1,000 carefully selected, *diverse* examples is sufficient to produce a well-aligned model.

The paper's data curation strategy follows directly from this hypothesis. The 1,000 LIMA examples are drawn from a wide range of domains and tasks — coding, factual Q&A, creative writing, advice, and open-ended conversation — specifically because diversity of task types is treated as the mechanism that generalizes alignment. Zhou et al. explicitly argue that their diversity-first selection "covers a wide range of topics and formats" in order to produce broad capability (§4, "Selection Criteria").

---

## Why I Disagree

The Superficial Alignment Hypothesis is plausible when the target behavior is already latent in the base model's pretraining distribution. It breaks down when the target behavior requires suppressing a strong prior or learning a constraint that is domain-specific and absent from the pretraining corpus.

Consider Tenacious's banned-phrase constraint. The base model has been trained on billions of tokens of corporate writing that contain exactly the phrases Tenacious prohibits: "leverage," "synergy," "just checking in," "touching base," "I hope this finds you well." These phrases are common, not rare. The base model's prior for professional email generation is biased *toward* them. Fine-tuning on a LIMA-style diverse dataset would include a few good examples that avoid these phrases, but also many examples from coding, factual Q&A, and creative writing that are irrelevant to the constraint — and the "diversity" signal would dilute the few examples that actually target the failure mode.

Week 10 traces make this concrete. In trace_107, the agent used "leverage," "synergy," and "synergising" in a discovery follow-up despite the system prompt stating Tenacious's voice guidelines. In trace_315, "just checking in" appeared in both the subject and body of a follow-up despite an explicit constraint. The failure taxonomy shows tone_drift (38%) and signal_missing (29%) together account for 67% of all Week 10 failures. These are not gaps in general language capability — the base model can clearly write fluent professional email. They are failures of targeted constraint adherence, and targeted constraint adherence is precisely what diversity-first curation does not optimize for.

The LIMA approach would produce a fine-tuned model that improves on general format compliance, instruction following, and response length. It would not reliably suppress a phrase that appears thousands of times in the base model's training distribution just because a handful of the 1,000 curated examples happen not to use it.

---

## What Should Have Been Done Differently

The paper would have been stronger with a scoped claim: diversity-first curation works when alignment targets behaviors already latent in the model. For cases where fine-tuning must suppress a strong learned prior, density in the failure-mode space outperforms diversity across task types.

Practically, this means that for the Tenacious SFT run, the training data should be *over-represented* in tone_drift and signal_missing examples (the two modes that account for 67% of Week 10 failures), rather than evenly spread across task types. A LIMA-style 50-example diverse sample would likely leave tone compliance largely unchanged. A 200-example targeted sample — all drawn from email_outreach and follow_up tasks, all demonstrating signal-led openers and zero banned phrases — should produce measurable improvement on the specific failure modes the evaluation is designed to detect.

This is the design rationale for using trace_derived and programmatic tasks as the backbone of the Tenacious-Bench training split rather than a general diversity-maximizing sample.

---

*~520 words | Week 11 evidence: trace_107, trace_315, failure taxonomy (tone_drift 38%, signal_missing 29%)*
