"""Conditional, reached-position acceptance from uncapped accepted-run counts."""
from __future__ import annotations

import math


def validate_histogram(histogram, *, steps=None, accepted=None, draft_length=15):
    if not isinstance(histogram, list) or not histogram:
        raise ValueError("Missing accepted-draft histogram")
    if len(histogram) > draft_length + 1:
        raise ValueError("Accepted run exceeds the configured draft length")
    if any(type(value) is not int or value < 0 for value in histogram):
        raise ValueError("Histogram must contain nonnegative integer counts")
    if steps is not None and sum(histogram) != steps:
        raise ValueError("Acceptance histogram does not match verifier-call count")
    if accepted is not None and sum(i * n for i, n in enumerate(histogram)) != accepted:
        raise ValueError("Acceptance histogram does not match accepted-draft count")
    return histogram + [0] * (draft_length + 1 - len(histogram))


def acceptance_summary(rows, draft_length=15):
    measured = [row for row in rows if "accepted_drafts_histogram" in row]
    if not measured:
        return None
    if len(measured) != len(rows):
        raise ValueError("Cannot pool measured and unmeasured acceptance histograms")
    pooled = [0] * (draft_length + 1)
    for row in measured:
        counts = validate_histogram(row["accepted_drafts_histogram"],
            steps=row["verify_calls"], accepted=row["accepted_drafts"],
            draft_length=draft_length)
        pooled = [a + b for a, b in zip(pooled, counts)]
    steps = sum(pooled)
    positions = []
    for position in range(1, draft_length + 1):
        reached = sum(pooled[position - 1:])
        accepted = sum(pooled[position:])
        rate = accepted / reached if reached else None
        interval = None
        if reached:
            z = 1.959963984540054
            denominator = 1 + z * z / reached
            center = (rate + z * z / (2 * reached)) / denominator
            radius = z * math.sqrt(rate * (1 - rate) / reached +
                                  z * z / (4 * reached * reached)) / denominator
            interval = [max(0, center - radius), min(1, center + radius)]
        positions.append(dict(position=position, reached=reached, accepted=accepted,
            conditional_acceptance=rate, wilson95_descriptive=interval,
            prefix_survival=accepted / steps if steps else None))
    return dict(accepted_drafts_histogram=pooled, verifier_calls=steps,
        uncapped_tpf=1 + sum(i * n for i, n in enumerate(pooled)) / steps if steps else None,
        positions=positions,
        definition="P(A >= i | A >= i-1), where A excludes the verifier bonus token",
        censoring="Full final-block outcomes retained before output-budget truncation",
        uncertainty="Wilson intervals are descriptive; tokens within a request are correlated")
