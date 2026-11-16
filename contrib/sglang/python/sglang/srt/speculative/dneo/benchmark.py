"""Portable matched-cohort benchmark with honest request/decode boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .loading import load_checkpoint, resolve_model
from .runtime.dneo_production import activate_production_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sglang", "transformers"), required=True)
    parser.add_argument("--method", choices=("dneo", "ar"), default="dneo")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--target-revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", choices=(0., 1.), type=float, default=1.)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 32, 64])
    parser.add_argument("--engine-capacity", type=int, default=256)
    parser.add_argument("--mem-fraction-static", type=float, default=.60)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--matched-requests", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.environ.get("DNEO_REFERENCE_MATCHED_ATTENTION", "0") == "1":
        parser.error("reference-matched correctness mode must not be reported as optimized performance")
    if args.output.exists():
        raise FileExistsError("use a new output path")
    if any(batch < 1 for batch in args.batches) or args.max_new_tokens < 2:
        raise ValueError("positive batches and output budget >= 2 required")
    if not 0 < args.mem_fraction_static < 1:
        raise ValueError("memory fraction must be between zero and one")
    if args.backend == "sglang" and args.engine_capacity <= max(args.batches):
        raise ValueError("scheduler capacity must exceed the submitted cohort")
    bank = json.loads(args.prompt_bank.read_text())
    records = bank["prompts"]
    if not records or any(not 0 < len(row["token_ids"]) <= 256 for row in records):
        raise ValueError("nonempty pretokenized prompt bank with <=256-token prompts required")
    if any(len(row["token_ids"]) + args.max_new_tokens > 1024 for row in records):
        raise ValueError("benchmark context cap exceeded")
    activate_production_runtime()
    from .configuration_dneo import DNeoConfig
    if args.method == "dneo":
        if not args.draft_model:
            parser.error("DNeo requires --draft-model")
        root = resolve_model(args.draft_model)
        config = DNeoConfig.from_pretrained(root)
        if config.block_size != 16:
            raise ValueError("this matched benchmark recipe requires block size 16")
    else:
        root = None
        config = DNeoConfig(target_model_name_or_path=args.target_model, target_revision=args.target_revision)
    manifest = dict(backend=args.backend, method=args.method, temperature=args.temperature,
        draft_temperature=1., batches=args.batches, engine_capacity=args.engine_capacity,
        memory_fraction_static=args.mem_fraction_static,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup_tokens,
        prompt_bank_sha256=hashlib.sha256(args.prompt_bank.read_bytes()).hexdigest(), matched_requests=args.matched_requests,
        target=config.target_model_name_or_path, target_revision=config.target_revision,
        hardware=torch.cuda.get_device_name(), full_cohorts_only=True, conditions=[])
    engine = None
    startup = time.perf_counter()
    if args.backend == "transformers":
        from .runtime.hf_cohort import CohortHFTarget, DNeoCohortDraft
        from .runtime.cohort_benchmark import generate_cohort
        if args.method == "dneo":
            target, model, _, _, _ = load_checkpoint(root)
        else:
            from transformers import AutoModelForCausalLM
            target = AutoModelForCausalLM.from_pretrained(config.target_model_name_or_path, revision=config.target_revision,
                        dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
        native = CohortHFTarget(target, max_batch=max(args.batches), capacity=1040,
                               feature_layers=model.config.target_layer_ids if args.method == "dneo" else ())
        native.prepare(args.batches, widths=(1, 16))
        draft = DNeoCohortDraft(model, 1040, max(args.batches)) if args.method == "dneo" else None
    else:
        from sglang import Engine
        from .runtime.stream_timing import streamed_generate, warmup_prefill
        engine = Engine(model_path=config.target_model_name_or_path, revision=config.target_revision,
            speculative_algorithm="DFLASH" if args.method == "dneo" else None,
            speculative_draft_model_path=str(root) if args.method == "dneo" else None,
            speculative_num_draft_tokens=16 if args.method == "dneo" else None,
            speculative_use_rejection_sampling=args.method == "dneo", tp_size=1, dtype="bfloat16",
            trust_remote_code=True, skip_tokenizer_init=True, context_length=1024,
            disable_radix_cache=True, cuda_graph_backend_prefill="disabled",
            max_running_requests=args.engine_capacity, cuda_graph_max_bs_decode=max(args.batches),
            attention_backend="fa3", mem_fraction_static=args.mem_fraction_static, enable_metrics=True)
    try:
        manifest["startup_seconds"] = time.perf_counter() - startup
        for batch in args.batches:
            total = min(len(records), args.matched_requests) // batch * batch
            if not total:
                raise ValueError("not enough prompts for a full cohort")
            rows = []
            for start in range(0, len(records), batch):
                group = records[start:start + batch]
                if engine is None:
                    if args.warmup_tokens:
                        generate_cohort(native, draft, group, args.warmup_tokens, args.temperature, args.seed + start)
                    hist = [[0] * 16 for _ in group] if draft else None
                    calls, elapsed, decode, ttft = generate_cohort(native, draft, group, args.max_new_tokens,
                        args.temperature, args.seed + start, acceptance_histograms=hist)
                    tokens = len(group) * args.max_new_tokens
                else:
                    ids = [row["token_ids"] for row in group]
                    params = [dict(temperature=args.temperature, top_p=1., top_k=-1,
                                   max_new_tokens=args.max_new_tokens, ignore_eos=True,
                                   sampling_seed=args.seed + start + index) for index in range(len(group))]
                    if args.warmup_tokens:
                        warmup_prefill(engine, ids, params, args.warmup_tokens)
                    results, elapsed, decode, first = streamed_generate(engine, ids, params)
                    tokens = sum(row["meta_info"]["completion_tokens"] for row in results)
                    calls = [row["meta_info"].get("spec_verify_ct", 0) for row in results]
                    hist = [row["meta_info"].get("spec_correct_drafts_histogram") for row in results]
                    ttft = sum(first) / len(first)
                rows.append(dict(start=start, benchmarks=[row["benchmark"] for row in group],
                    tokens=tokens, verify_calls=calls, accepted_drafts_histograms=hist,
                    request_seconds=elapsed, decode_seconds=decode, mean_ttft_seconds=ttft,
                    cohort_size=len(group), timing_qualified=len(group) == batch and start + len(group) <= total,
                    request_tps=tokens / elapsed, decode_tps=(tokens - len(group)) / decode,
                    tpf=tokens / sum(calls) if sum(calls) else None))
            qualified = [row for row in rows if row["timing_qualified"]]
            manifest["conditions"].append(dict(batch=batch, rows=rows,
                request_tps=sum(row["tokens"] for row in qualified) / sum(row["request_seconds"] for row in qualified),
                decode_tps=sum(row["tokens"] - batch for row in qualified) / sum(row["decode_seconds"] for row in qualified)))
            args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
