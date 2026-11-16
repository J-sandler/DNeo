"""Native DNeo loading; reuse upstream DFLASH verification, not its drafter."""
import time
import hashlib
import json
from pathlib import Path

import torch
from sglang.srt.distributed import get_tp_group
from sglang.srt.runtime_context import get_schedule
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2, SpecTpSync

from dneo.loading import load_draft
from dneo.runtime.dneo_production import activate_production_runtime
from dneo.runtime.engine import DNeoEngine, _set_inference_block_size
from dneo.runtime.native_hooks import install_hooks


class DNeoWorker(DNeoEngine, DFlashWorkerV2):
    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        activate_production_runtime()
        self._verify_runtime_fingerprint()
        if server_args.tp_size != 1 or not server_args.disable_radix_cache:
            raise ValueError("DNeo currently requires TP=1 and --disable-radix-cache")
        if server_args.cuda_graph_backend_prefill != "disabled":
            raise ValueError("DNeo requires --cuda-graph-backend-prefill disabled")
        if server_args.speculative_draft_window_size is not None:
            raise ValueError("DNeo does not truncate draft prefix windows")
        BaseSpecWorker.__init__(self)
        self.server_args, self.gpu_id, self.ps, self.nccl_port = server_args, gpu_id, ps, nccl_port
        self._target_worker, self.model_runner = target_worker, target_worker.model_runner
        self.device, self.page_size = target_worker.device, get_schedule().page_size
        self._draft_worker = None
        self._need_mamba_verify_commit = self.use_compact_draft_cache = self._is_domino = False
        self._tp_sync = SpecTpSync(get_tp_group())
        start = time.perf_counter()
        model = load_draft(server_args.speculative_draft_model_path,
                           revision=server_args.speculative_draft_model_revision)
        if server_args.revision != model.config.target_revision:
            raise ValueError(f"pin the compatible target with --revision {model.config.target_revision}")
        target_config = self.model_runner.model_config.hf_config
        if (target_config.model_type != "qwen3"
                or target_config.num_hidden_layers != model.config.num_target_layers
                or target_config.hidden_size != model.config.hidden_size):
            raise ValueError("DNeo checkpoint is incompatible with the selected target architecture")
        self.block_size = int(server_args.speculative_num_draft_tokens or model.block_size)
        _set_inference_block_size(model, self.block_size)
        self.speculative_num_draft_tokens = self.block_size
        model.bind_target(self.model_runner.model)
        if model.embed_tokens.weight.dtype != torch.bfloat16:
            raise ValueError("the optimized DNeo release requires a BF16 target vocabulary")
        self.capacity = int(server_args.context_length) + self.block_size
        self._dneo_preloaded_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self._dneo_weight_load_seconds = time.perf_counter() - start
        self._block_pos_offsets = torch.arange(self.block_size, device=self.device)
        for name in ("_draft_block_ids_buf", "_draft_block_positions_buf", "_draft_block_tokens_buf",
                     "_draft_verify_out_cache_loc_buf", "_draft_block_end_buf", "_draft_seq_lens_cpu_buf",
                     "_accept_len_buf"):
            setattr(self, name, None)
        self._use_triton_prepare_block = True
        self._accept_bonus_buffer_cap = self._accept_bonus_buffer_slot = 0
        self._commit_lens_bufs, self._bonus_id_bufs, self._out_tokens_bufs, self._new_seq_lens_bufs = [], [], [], []
        self.draft_model = model
        install_hooks(self, model, {"architecture": model.config.to_dict()})

    @property
    def preloaded_weights_bytes(self):
        return self._dneo_preloaded_bytes

    @property
    def weight_load_time(self):
        return self._dneo_weight_load_seconds

    @property
    def spec_v2_attn_backends(self):
        return (self.model_runner.attn_backend,)

    def alloc_memory_pool(self, **kwargs):
        # DNeo's indexed feature KV buffers are allocated by the shared engine.
        pass

    def init_attention_backends(self):
        pass

    @staticmethod
    def _verify_runtime_fingerprint():
        import sglang
        root = Path(sglang.__file__).resolve().parent
        manifest = json.loads(Path(__file__).with_name("sglang_manifest.json").read_text())
        for relative, expected in manifest["files"].items():
            path = root / relative.removeprefix("python/sglang/")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"SGLang runtime differs from the release fingerprint: {relative}")
