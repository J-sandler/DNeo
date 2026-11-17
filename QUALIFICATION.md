# Shared-Router Update

Version 0.1.0rc6 supports one checkpoint-defined router shared across all draft
layers and KV heads, while retaining old per-layer checkpoint compatibility.
The optimized v5 kernels, head, attention, and acceptance policy are unchanged.
The source bundle and in-tree SGLang contribution use the same model code.

The adapted checkpoint's matched online TPF was 4.2109 versus 4.3024 for the
previous checkpoint. Default selection was explicitly approved despite the
2.13% TPF decrease. Earlier serving throughput numbers concern the previous
weights and must not be presented as new measurements of this checkpoint.

Numerical/storage compatibility and GPU decode results for this update are
recorded in SHARED_ROUTER_QUALIFICATION.json when qualification has completed.
This update does not inherit the prior release's performance attestation.
