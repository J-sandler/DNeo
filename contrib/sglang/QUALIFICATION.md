# Release Qualification

This release refreshes the documentation of engineering-qualified rc5.
All inference Python/JSON files, integration patches, dependency locks, model
configuration, and learned tensors are byte-identical to that qualified release.
Archive and wheel metadata hashes changed; the wheel has build tag 1docs.
No new GPU execution is claimed for this documentation-only refresh.

The original qualification passed 60 numerical fixtures, generated-token parity
over 1,280 requests / 655,360 tokens, and 80 paired H100 performance cells.
The later fresh-source quickstart passed 20,480 exact acceptance-count/histogram
request comparisons; throughput differed from its references by at most 2.83%.
Exactness applies to the specified fixtures and counts, not arbitrary hardware,
all possible inputs, or wall-clock time. See the public qualification reports.

Supported performance scope: H100 80GB, TP=1, BF16, batches 1/4/16/32/64,
block size 16, total context 1024, target temperature 0/1 and draft temperature 1.
No vLLM, TP>1, tree, other-GPU, or unbounded-context performance claim is made.
The matched benchmark prompt bank is not bundled; different prompts change TPF.

Code retains MIT. The author selected Apache-2.0 for the draft weights;
upstream notices remain intact. This records the release license choice, not
an independent legal opinion about ownership or third-party rights.
