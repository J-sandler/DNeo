"""Matched serving timing with observable first-token and cohort boundaries."""
import time
from collections import defaultdict
from statistics import median


def warmup_prefill(engine, input_ids, sampling_params, max_new_tokens,
                   clock=time.perf_counter):
    if max_new_tokens < 1:
        raise ValueError('Prefill warmup needs at least one generated token')
    params = [{**row, 'max_new_tokens': max_new_tokens} for row in sampling_params]
    final, elapsed, _, first = streamed_generate(engine, input_ids, params, clock)
    return dict(requests=len(input_ids), prompt_lengths=[len(row) for row in input_ids],
                max_new_tokens=max_new_tokens, elapsed_seconds=elapsed,
                client_ttft_seconds=first,
                server_ttft_seconds=[row['meta_info'].get('first_token_latency') for row in final])


def streamed_generate(engine, input_ids, sampling_params, clock=time.perf_counter):
    started = clock()
    count = len(input_ids)
    first = [None] * count
    final = [None] * count
    for chunk in engine.generate(input_ids=input_ids, sampling_params=sampling_params, stream=True):
        now = clock()
        index = int(chunk.get('index', 0))
        if not 0 <= index < count:
            raise ValueError('Invalid streaming request index')
        if chunk['meta_info'].get('completion_tokens', 0) > 0 and first[index] is None:
            first[index] = now
        final[index] = chunk
    ended = clock()
    if any(value is None for value in first) or any(value is None for value in final):
        raise ValueError('Missing streamed requests or first tokens')
    # Counting every continuation token requires retaining the interval in
    # which early requests decode while later requests are still prefilling.
    decode = ended - min(first)
    if decode <= 0:
        raise ValueError('No measurable cohort decode interval')
    return final, ended - started, decode, [value - started for value in first]


def corrected_cohort_decode_rows(rows):
    """Recover the earliest-first-token boundary without modifying raw data."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["batch"]].append(dict(row))
    corrected = []
    for group in groups.values():
        first = min(row.get("client_ttft_seconds", row["ttft_seconds"]) for row in group)
        elapsed = sum(row["client_elapsed_seconds"] for row in group)
        decode = elapsed - first
        tokens = sum(row["completion_tokens"] - 1 for row in group)
        if decode <= 0 or tokens <= 0:
            raise ValueError("No measurable corrected cohort decode interval")
        for row in group:
            row["decode_seconds"] = decode * (row["completion_tokens"] - 1) / tokens
            row["batch_decode_seconds"] = decode
            corrected.append(row)
    return corrected


def cohort_admission_summary(rows):
    """Flag late first-token outliers; this does not infer exact GPU occupancy."""
    if not rows:
        raise ValueError('An admission audit needs a nonempty cohort')
    first = [row.get('client_ttft_seconds', row['ttft_seconds']) for row in rows]
    elapsed = sum(row['client_elapsed_seconds'] for row in rows)
    decode = elapsed - min(first)
    if decode <= 0:
        raise ValueError('No measurable cohort decode interval')
    typical = median(first)
    threshold = typical + max(.5, .25 * decode)
    return dict(requests=len(rows), elapsed_seconds=elapsed, decode_seconds=decode,
                first_token_min_seconds=min(first), first_token_median_seconds=typical,
                first_token_max_seconds=max(first),
                late_first_token_requests=sum(value > threshold for value in first),
                admission_skew_detected=any(value > threshold for value in first))
