# Cached-teacher training

For the released 20K cache, fixed Hub revision, official-data conversion,
initialization checks, and complete training/export commands, start with
[README sections 4–8](../README.md#4-download-the-training-bundle) or the
[Chinese workflow](../README_zh.md#4-下载并校验训练路由缓存).
The Git repository's `configs/` files are generic templates; the released
bundle supplies dataset-specific YAMLs using documented environment variables.

Install the pinned runtime requirements, then `requirements-train.txt` and
the project in editable mode. Use a separate environment. The tested LeRobot
API is v2.1-era; a current unrelated LeRobot release is not interchangeable.

Required inputs (not stored in Git; routes and normalization are available on
[Hugging Face](https://huggingface.co/xhx0327/WorldLA-100k-decay-full/tree/main/training/stage_routes_20k)):

1. A LeRobot v2.1 robot dataset with three image keys
   `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist`,
   `observation.state` (14), and `action` (14), plus task/prompt metadata.
2. Matching `<asset_dir>/<asset_id>/norm_stats.json`. Do not reuse stats from
   a different dataset or change the state/action coordinate convention.
3. The pi0.5 base `params/` checkpoint and PaliGemma tokenizer.
4. Stage-route NPZ caches and a manifest; format below.

## Route caches

Each manifest row contains `sample_id`, `global_index`, `episode_index`,
`frame_index`, `split` (`train`, `val`, or `test`), `family`, and
`stage_route_npz`. The NPZ has `stage_route_target`, shape `[4,768]` for three
16x16 view grids. Its final dimension sums to one for each stage. Relative
paths are resolved against the manifest, not the current working directory.
`train_stage_priors.npz` contains `global_stage_mean` computed from train only.

No episode may appear in multiple splits. All episodes must be represented
in the manifest for full-frame action-stage indexing; indices/frame positions
must match the underlying LeRobot dataset exactly. Do not reorder datasets
after generating caches. The historical split is stratified by task and is
not a held-out-task generalization protocol.

## Convert existing teacher extraction caches

```bash
worldla build-routes --teacher-manifest data/teacher/teacher_manifest.jsonl \
  --output-dir data/stage_routes
```

Input NPZ: `shared_k_route_abs_ta` (or `shared_k_route_probability_ta`), shape
`[denoise,action,120]`. Input manifest needs the row fields above and
`teacher_npz`. The original teacher uses a 12x10 composite spatial grid:
head view is top 8x10; lower left/right wrists are 4x5 each. These are mapped
separately to three 16x16 student grids with per-view mass preserved.
Teacher action positions are divided into four relative bins, denoising
reads are aggregated, and each stage is independently normalized.
The student's 50 action positions use `floor(index*4/50)` for message selection.

This conversion does **not** discover shared-K directions or run Fast-WAM.
The raw extractor depends on the separately validated Fast-WAM instrumentation
and discovery artifacts; it is not yet included in this standalone project.
Do not substitute an arbitrary attention heatmap and call it the same teacher.

`worldla build-anchors` prepares task-stratified anchors from an already
converted dataset. Dataset conversion and normalization must be completed
with tools matching the dataset schema; they are not silently performed here.

## Two training stages

`configs/router.yaml` matches the main 60k router budget. `configs/action_100k.yaml`
matches the scheduled main 100k action budget. Both are user-editable templates.
Do not mistake the teacher-ablation 20k/40k budgets for the main checkpoint.

- Router stage: backbone frozen; train router from cached distribution KL.
- Action stage: router frozen; update LoRA branches in both the VLM and action
  expert, plus adapter/gates, from
  flow-matching action loss over all frames of train episodes.
- Validation uses held-out episode anchors, not simulator rollouts.
- Checkpoints are parameter-only. `--resume-model-dir` is a warm start that
  resets optimizer moments; it is not a full-state training resume.

The source also retains explicitly named teacher-ablation conditions. R2
requires genuine cached WAM hidden features, not spatial route maps. R4 uses
fresh sample/step-dependent stage derangements, not a learnable fixed shuffle.
No matching-budget ablation results are fabricated in this source release.

## Short end-to-end training check

Before a full training run, use the installed package with real local data:

```bash
python scripts/smoke_training.py \
  --lerobot-root data/lerobot --dataset-repo-id robotwin_demo_clean \
  --asset-dir data/assets --asset-id robotwin_demo_clean \
  --route-manifest data/stage_routes/stage_route_manifest.jsonl \
  --init-params data/pi05_base/params \
  --tokenizer-path data/paligemma_tokenizer.model \
  --gpus 0,1,2,3 --batch-size 4 --steps 4 \
  --output outputs/train_smoke_01
```

The helper refuses existing output directories and GPUs with 512 MiB or more
already in use. It runs Router pretraining, restores that Router for the action
stage, saves an intermediate and a final action checkpoint, and restores the
saved parameters on CPU. Verification requires finite losses/nonzero gradients,
updated LoRA/adapter leaves, an unchanged pretrained Router during action
training, and no dense-base leaves in the adaptation checkpoint. Sampling and
validation use real dataset records, not synthetic model inputs; validation is
limited to one batch per stage for this smoke test.

All new files, dataset/JAX caches, logs, and small test checkpoints are written
under the requested output. The first run can spend time and disk space building
the local dataset cache. The helper is offline and does not download data,
overwrite existing checkpoints, publish artifacts, or establish full-budget
training convergence.
