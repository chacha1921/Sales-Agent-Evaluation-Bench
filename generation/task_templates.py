"""Shared RUBRICS dict and make_task helper used by all generation scripts."""
from datetime import date

TODAY = str(date.today())

RUBRICS = {
    "email_outreach": {
        "signal_grounding":        {"weight": 0.30, "checker": "signal_grounding_fn",   "description": "References ≥1 verifiable signal from the context."},
        "tone_compliance":         {"weight": 0.25, "checker": "tone_checker_fn",       "description": "Scores on 5 Tenacious tone markers (LLM-judge, T=0)."},
        "banned_phrase_absent":    {"weight": 0.15, "checker": "banned_phrase_fn",      "description": "Zero phrases from banned_phrases.txt."},
        "cta_present":             {"weight": 0.15, "checker": "cta_checker_fn",        "description": "Calendar link or booking CTA present."},
        "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn",         "description": "Within constraint word limit."},
    },
    "follow_up": {
        "signal_grounding":        {"weight": 0.25, "checker": "signal_grounding_fn",   "description": "References new or existing signal."},
        "tone_compliance":         {"weight": 0.25, "checker": "tone_checker_fn",       "description": "Non-pushy follow-up tone."},
        "banned_phrase_absent":    {"weight": 0.25, "checker": "banned_phrase_fn",      "description": "No 'just checking in' or equivalent."},
        "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn",         "description": "Under word limit."},
        "cta_present":             {"weight": 0.10, "checker": "cta_checker_fn",        "description": "Specific next step."},
    },
    "discovery_response": {
        "signal_grounding":        {"weight": 0.25, "checker": "signal_grounding_fn",   "description": "References discovery signal."},
        "tone_compliance":         {"weight": 0.25, "checker": "tone_checker_fn",       "description": "Tenacious voice compliance."},
        "banned_phrase_absent":    {"weight": 0.20, "checker": "banned_phrase_fn",      "description": "Zero banned phrases."},
        "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn",         "description": "Within word limit."},
        "cta_present":             {"weight": 0.15, "checker": "cta_checker_fn",        "description": "Specific next step."},
    },
    "objection_handling": {
        "signal_grounding":        {"weight": 0.20, "checker": "signal_grounding_fn",   "description": "References context signal."},
        "objection_acknowledged":  {"weight": 0.30, "checker": "objection_ack_fn",      "description": "Acknowledges objection before pivoting."},
        "tone_compliance":         {"weight": 0.25, "checker": "tone_checker_fn",       "description": "Tenacious voice."},
        "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn",         "description": "Under word limit."},
        "banned_phrase_absent":    {"weight": 0.10, "checker": "banned_phrase_fn",      "description": "Zero banned phrases."},
    },
    "closing": {
        "tone_compliance":         {"weight": 0.30, "checker": "tone_checker_fn",       "description": "Tenacious voice at close."},
        "banned_phrase_absent":    {"weight": 0.20, "checker": "banned_phrase_fn",      "description": "Zero banned phrases."},
        "cta_present":             {"weight": 0.25, "checker": "cta_checker_fn",        "description": "Specific next step or calendar link."},
        "word_count_within_limit": {"weight": 0.25, "checker": "word_count_fn",         "description": "Concise close."},
    },
    "email_outreach_no_pricing": {
        "signal_grounding":        {"weight": 0.30, "checker": "signal_grounding_fn",   "description": "References ≥1 verifiable signal."},
        "pricing_absent":          {"weight": 0.20, "checker": "pricing_mention_fn",    "description": "No pricing on first touch."},
        "tone_compliance":         {"weight": 0.20, "checker": "tone_checker_fn",       "description": "Tenacious tone markers."},
        "banned_phrase_absent":    {"weight": 0.15, "checker": "banned_phrase_fn",      "description": "Zero banned phrases."},
        "cta_present":             {"weight": 0.10, "checker": "cta_checker_fn",        "description": "CTA present."},
        "word_count_within_limit": {"weight": 0.05, "checker": "word_count_fn",         "description": "Within word limit."},
    },
}

THRESHOLD = {
    "email_outreach": 3.5,
    "follow_up": 3.5,
    "discovery_response": 3.5,
    "objection_handling": 3.0,
    "closing": 3.5,
    "email_outreach_no_pricing": 3.5,
}


def make_task(tid, authoring_mode, source_traces, difficulty, context, task_type,
              constraints, segment, failure_tag, adv_weight, signal_source,
              signal_time_window, generation_model="programmatic", ref_output="",
              seed_id=None):
    rubric_key = task_type if task_type in RUBRICS else "email_outreach"
    return {
        "task_id": f"TB-{tid:04d}",
        "split": "",
        "authoring_mode": authoring_mode,
        "source_trace_ids": source_traces,
        "difficulty": difficulty,
        "input": {
            "context": context,
            "task_type": task_type,
            "constraints": constraints,
        },
        "candidate_output": "",
        "ground_truth": {
            "type": "rubric",
            "dimensions": RUBRICS[rubric_key],
            "aggregate": "weighted_sum",
            "scale": "0.0 to 5.0",
            "passing_threshold": THRESHOLD.get(task_type, 3.5),
        },
        "metadata": {
            "tenacious_segment": segment,
            "failure_mode_tag": failure_tag,
            "adversarial_weight": adv_weight,
            "generation_model": generation_model,
            "judge_model": None,
            "judge_scores": None,
            "signal_source": signal_source,
            "signal_time_window": signal_time_window,
            "seed_id": seed_id,
            "created_at": TODAY,
            "reference_output": ref_output,
        },
    }
