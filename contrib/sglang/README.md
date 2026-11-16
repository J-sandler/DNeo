# DNeo SGLang Integration

Use the pinned build and ordinary launch_server command in QUICKSTART.md.
DFLASH selects the upstream scheduling family; DNeo config selects the
native DNeoWorker. Unmodified upstream SGLang is not supported.
The three patches include runtime and text-only dependency compatibility
changes that should be reviewed/split before an upstream PR. Native serving
uses the in-tree payload, not the standalone dneo-inference wheel.

The exact source archive used by that quickstart is a companion release asset.

Code: https://github.com/J-sandler/DNeo
Weights: https://huggingface.co/J-sandler/Qwen3-4B-DNeo
