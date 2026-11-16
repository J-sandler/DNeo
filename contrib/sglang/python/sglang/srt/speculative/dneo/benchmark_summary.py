"""Compare matched public benchmark runs without mixing timing boundaries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .runtime.acceptance import acceptance_summary


def compare(spec, ar):
    if any(document.get("cublas_workspace_config") != ":4096:8" for document in (spec, ar)):
        raise ValueError("unmatched or unqualified benchmark setting: cublas_workspace_config")
    if spec.get("memory_fraction_static") != ar.get("memory_fraction_static"):
        raise ValueError("unmatched benchmark setting: memory_fraction_static")
    for key in ("backend", "temperature", "prompt_bank_sha256", "matched_requests", "max_new_tokens", "target", "target_revision", "engine_capacity"):
        if spec[key] != ar[key]:
            raise ValueError(f"unmatched benchmark setting: {key}")
    if spec["method"] != "dneo" or ar["method"] != "ar":
        raise ValueError("expected DNeo and AR results")
    baseline = {row["batch"]: row for row in ar["conditions"]}
    results = []
    for condition in spec["conditions"]:
        batch = condition["batch"]
        reference = baseline[batch]
        groups = defaultdict(list)
        for cohort in condition["rows"]:
            for name, calls, histogram in zip(cohort["benchmarks"], cohort["verify_calls"], cohort["accepted_drafts_histograms"]):
                row = dict(verify_calls=calls, accepted_drafts_histogram=histogram,
                           accepted_drafts=sum(index * count for index, count in enumerate(histogram)))
                groups[name].append(row)
        results.append(dict(batch=batch, request_speedup=condition["request_tps"] / reference["request_tps"],
            decode_speedup=condition["decode_tps"] / reference["decode_tps"],
            request_tps=condition["request_tps"], decode_tps=condition["decode_tps"],
            per_benchmark={name: dict(prompts=len(rows),
                tpf=len(rows) * spec["max_new_tokens"] / sum(row["verify_calls"] for row in rows),
                conditional_acceptance=acceptance_summary(rows)) for name, rows in groups.items()}))
    return dict(backend=spec["backend"], temperature=spec["temperature"], conditions=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dneo", type=Path, required=True)
    parser.add_argument("--ar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.dneo.read_text()), json.loads(args.ar.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
