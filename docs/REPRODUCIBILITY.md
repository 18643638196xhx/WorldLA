# Reproducibility boundaries

This is a source refactor, not a new benchmark. Preserve existing checkpoint
and experiment provenance. A passing unit test is not task-success evidence.

The main scheduled action run uses 100k updates, batch 32, peak LR 2e-4,
3k warmup, plateau until 50k, then cosine decay to 2e-5. The main router budget
is 60k. Metadata records the actual model config, training condition, split,
checkpoint paths, schedule, random seed, parameter counts, and warm-start state.

Complete checkpoints retain 50-step action horizon, 10-step execution blocks,
ten denoising iterations, and old/new arm weights 0.75/0.25 by default.
Inference normalization and discrete gripper handling must remain unchanged.

`original` retains separate LoRA factors and the original bf16 computation.
`merged` folds factors into dense matrices; the existing export retains folded
matrices in float32 to avoid another rounding. This explains the larger file
size. LoRA folding can change results despite mathematical real-arithmetic
equivalence. Do not assume unchanged task SR from fixed-input closeness.

The independent project loads full release folders through its own packaged
model, not by importing Python files from a downloaded checkpoint. Vendored
model equations must match the snapshot recorded in `SOURCE_MANIFEST.json`.
Run unit tests, build/install a wheel, test from a relocated directory, then
compare fixed-input actions/routes and finally simulator rollouts before
asserting full reproduction of reported results.
