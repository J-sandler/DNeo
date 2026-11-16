"""Native feature capture hooks, without the legacy target allocation."""
import gc
import os
import torch

def install_hooks(self, model, metadata):
    server_args, target_worker = self.server_args, self._target_worker
    self.dneo_metadata = metadata
    if not model.conditioner.per_layer_mofa:
        raise ValueError("native dNEO currently requires per-layer MoFA routing")
    self.dneo_model = model.cuda().eval().requires_grad_(False)
    gc.collect()
    self._mask_token_id = model.mask_token_id
    from dneo.runtime.feature_layout import feature_layer_ids
    self.feature_layer_ids = feature_layer_ids(model.conditioner)
    self._captured = [None] * len(self.feature_layer_ids)
    self._capture_handles = []
    for index, layer_id in enumerate(self.feature_layer_ids):
        layer = self.model_runner.model.model.layers[layer_id]
        def capture(module, inputs, index=index):
            self._captured[index] = inputs[0]
        self._capture_handles.append(layer.self_attn.qkv_proj.register_forward_pre_hook(capture))
    self.draft_temperature = float(os.environ.get("SGLANG_SPEC_DRAFT_TEMPERATURE_OVERRIDE", "1"))
    if self.draft_temperature <= 0:
        raise ValueError("native dNEO draft temperature must be positive")
    self._native_graphs_ready = False
    self._max_batch = int(server_args.max_running_requests)
    self._disable_graph = server_args.disable_cuda_graph
    self._request_slots = {}
    self._requests = {}
    self._active_slots = None
    self._current_batch = None
    self._request_id = None
    self._target_temperature = 1.0
    self._cache_routed_memory = os.environ.get("DNEO_NATIVE_CACHE_MOFA", "1") == "1"
    self._reuse_memory_kv = (
        os.environ.get("DNEO_REUSE_MEMORY_KV", "0") == "1"
        and bool(getattr(model.draft, "rejection_memory_kv_only", False))
        and getattr(model.conditioner, "rejection_router_logits", None) is None
    )
    self._profile_path = os.environ.get("SGLANG_DNEO_PROFILE")
    self._profile_cycles = 0
    self._profile_rows = []
    self._timings = None
    self._active_verify_features = None
    self._verify_feature_buffer = torch.empty(
        (self._max_batch * self.block_size, len(self._captured),
         self.dneo_model.embed_tokens.weight.shape[1]),
        device="cuda", dtype=torch.bfloat16,
    )
    if self._max_batch > 1:
        # This copy is recorded in each upstream target-verification graph.
        # A shared destination makes padded graph buckets transparent to dNEO.
        def capture_features(runner, output, forward_batch, num_tokens):
            torch.stack(tuple(self._captured), dim=1,
                        out=self._verify_feature_buffer[:num_tokens])
        self.model_runner.capture_tail_hooks.append(capture_features)
    original = target_worker.forward_batch_generation
    def forward_target(*args, **kwargs):
        output = self._timed("target", original, *args, **kwargs)
        if kwargs.get("is_verify", False):
            if self._max_batch > 1:
                if not output.can_run_cuda_graph:
                    count = output.logits_output.next_token_logits.shape[0]
                    torch.stack(tuple(x[:count] for x in self._captured), dim=1,
                                out=self._verify_feature_buffer[:count])
            else:
                self._active_verify_features = (
                    self._verify_features if output.can_run_cuda_graph else tuple(self._captured)
                )
        return output
    target_worker.forward_batch_generation = forward_target
