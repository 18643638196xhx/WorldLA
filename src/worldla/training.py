#!/usr/bin/env python3
"""Two-stage training for the four-stage pi0.5 Relation Router."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


from worldla.io import write_json
from worldla.runtime import configure_runtime
from worldla.data.stage_routes import CONDITIONS, ROUTE_CONDITIONS

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("router", "action"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--runtime-cache-dir",
        type=Path,
        help="Reuse an existing OpenPI/Hugging Face runtime cache.",
    )
    parser.add_argument("--config-name", default="worldla_pi05", choices=("worldla_pi05",))
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id", required=True,
        help="Optional local LeRobot repository name/path; defaults to the config repo.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path, required=True,
        help="Directory containing <asset-id>/norm_stats.json for dataset override.",
    )
    parser.add_argument("--asset-id", required=True, help="Normalization asset id for dataset override.")
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--global-feature-manifest", type=Path,
                        help="R2 pooled video hidden-feature targets; required only for its router stage.")
    parser.add_argument("--init-params", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--router-model-dir", type=Path)
    parser.add_argument(
        "--resume-model-dir",
        type=Path,
        help=(
            "Warm-start action training from a completed action-model directory. "
            "Model parameters are restored; optimizer state is restarted."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_plateau_cosine"),
        default="constant",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int)
    parser.add_argument("--lr-final", type=float)
    parser.add_argument(
        "--checkpoint-steps",
        default="",
        help="Comma-separated parameter-only evaluation snapshots; final step is always saved.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--router-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--max-validation-batches", type=int,
                        help="Optional smoke-test validation cap; leave unset for full validation.")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parameter_count(state):
    import jax
    return int(sum(np.prod(value.shape) for value in jax.tree.leaves(state)))


def training_filter(nnx, nnx_utils, *, stage: str, condition: str):
    """One source of truth for R1 joint learning vs frozen-teacher controls."""
    router = nnx_utils.PathRegex(".*stage_relation_router.*")
    adapter = nnx_utils.PathRegex(".*stage_message_adapter.*")
    lora = nnx_utils.PathRegex(".*lora.*")
    head = nnx_utils.PathRegex(".*global_feature_head.*")
    if stage == "router":
        return nnx.All(nnx.Param, nnx.Any(router, head))
    if condition == "base_lora":
        return nnx.All(nnx.Param, lora)
    if condition == "router_action_only":
        return nnx.All(nnx.Param, nnx.Any(lora, router, adapter))
    return nnx.All(nnx.Param, nnx.Any(lora, adapter))


def main():
    args = parse_args()
    if args.stage == "router" and args.condition not in ROUTE_CONDITIONS:
        raise ValueError(f"{args.condition} has no Router-pretraining stage")
    if args.steps <= 0 or args.batch_size <= 0 or args.validation_interval <= 0 or args.log_interval <= 0:
        raise ValueError("Steps, batch size, and validation/log intervals must be positive")
    if args.max_validation_batches is not None and args.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive")
    if args.stage == "action" and args.condition in ROUTE_CONDITIONS and args.router_model_dir is None:
        raise ValueError("--router-model-dir is required for routed action training")
    if args.condition == "router_action_only" and args.router_model_dir is not None:
        raise ValueError("R1 must start with a random Router; do not pass --router-model-dir")
    if args.stage == "router" and args.condition == "global_feature_distill" and args.global_feature_manifest is None:
        raise ValueError("R2 requires --global-feature-manifest containing genuine WAM hidden features")
    if args.resume_model_dir is not None and args.stage != "action":
        raise ValueError("--resume-model-dir is only supported for --stage action")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    configure_runtime(args.runtime_cache_dir, args.tokenizer_path)
    os.environ["LEROBOT_DATA_ROOT"] = str(args.lerobot_root.expanduser().resolve())
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")
    old_cwd = Path.cwd()

    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from openpi.models import gemma
    from openpi.shared import nnx_utils
    from openpi.training import config as training_config, sharding
    from openpi.training import utils as training_utils
    from openpi.training import weight_loaders

    from worldla.data.stage_routes import (
        FullEpisodeActionDataset,
        GlobalFeatureTargetStore,
        StageRouteOpenPiDataset,
        StageRouteTargetStore,
        create_batch_loader,
        create_native_dataset,
        split_batch,
        randomize_stage_targets,
    )
    from worldla.model import StageRelationRouterPi

    try:
        from worldla.config import create_training_config
        config = create_training_config(args)
        if not config.model.pi05:
            raise ValueError("Stage Relation Router requires a pi0.5 config")
        if args.dataset_repo_id is not None:
            if args.asset_dir is None or not args.asset_id:
                raise ValueError(
                    "--dataset-repo-id requires both --asset-dir and --asset-id"
                )
            base_data = config.data.base_config or training_config.DataConfig()
            base_data = dataclasses.replace(
                base_data,
                lerobot_data_root=str(args.lerobot_root.expanduser().resolve()),
            )
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(
                    config.data,
                    repo_id=str(args.dataset_repo_id),
                    assets=training_config.AssetsConfig(
                        assets_dir=str(args.asset_dir.expanduser().resolve()),
                        asset_id=str(args.asset_id),
                    ),
                    base_config=base_data,
                ),
            )
        store = StageRouteTargetStore(
            args.route_manifest,
            load_routes=args.condition not in {"base_lora", "router_action_only", "global_feature_distill"},
        )
        feature_store = None
        if args.stage == "router" and args.condition == "global_feature_distill":
            feature_store = GlobalFeatureTargetStore(args.global_feature_manifest, store.rows)
        if store.num_stages != 4:
            raise ValueError(f"Expected four stages, got {store.num_stages}")
        transformed, _ = create_native_dataset(config, args.lerobot_root)
        if args.stage == "router":
            train_dataset = StageRouteOpenPiDataset(
                transformed, store, split="train", condition=args.condition, feature_store=feature_store
            )
            train_mode = "global_visual_feature_anchors" if feature_store is not None else "four_stage_route_anchors"
        else:
            train_dataset = FullEpisodeActionDataset(transformed, store, split="train")
            train_mode = "all_frames_from_train_episodes"
        val_dataset = StageRouteOpenPiDataset(
            transformed, store, split="val", condition=args.condition, feature_store=feature_store
        )
        train_loader = create_batch_loader(
            train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed
        )
        val_loader = create_batch_loader(
            val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed
        )
        print(
            f"data stage={args.stage} condition={args.condition} mode={train_mode} "
            f"train={len(train_dataset)} val={len(val_dataset)}", flush=True
        )

        if args.fsdp_devices <= 0 or jax.device_count() % args.fsdp_devices:
            raise ValueError(
                f"Visible JAX devices ({jax.device_count()}) must be divisible by "
                f"--fsdp-devices={args.fsdp_devices}"
            )
        if args.batch_size % jax.device_count():
            raise ValueError("Global batch must be divisible by visible JAX devices")
        mesh = sharding.make_mesh(args.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        prefix_dim = int(gemma.get_config(config.model.paligemma_variant).width)
        suffix_dim = int(gemma.get_config(config.model.action_expert_variant).width)
        router_active = args.stage == "router" or args.condition != "base_lora"
        message_enabled = args.stage == "action" and args.condition != "base_lora"
        position_only = args.condition == "position_only"

        router_pattern = nnx_utils.PathRegex(".*stage_relation_router.*")
        adapter_pattern = nnx_utils.PathRegex(".*stage_message_adapter.*")
        global_head_pattern = nnx_utils.PathRegex(".*global_feature_head.*")
        lora_pattern = nnx_utils.PathRegex(".*lora.*")
        train_filter = training_filter(nnx, nnx_utils, stage=args.stage, condition=args.condition)
        deploy_filter = nnx.All(
            nnx.Param, nnx.Any(lora_pattern, router_pattern, adapter_pattern)
        )
        def create_model(rng):
            base_rng, wrapper_rng = jax.random.split(rng)
            return StageRelationRouterPi(
                config.model.create(base_rng),
                prefix_dim=prefix_dim,
                suffix_dim=suffix_dim,
                num_route_tokens=store.num_route_tokens,
                num_stages=store.num_stages,
                router_hidden_dim=args.router_hidden_dim,
                adapter_dim=args.adapter_dim,
                router_active=router_active,
                message_enabled=message_enabled,
                position_only=position_only,
                global_feature_dim=0 if feature_store is None else feature_store.feature_dim,
                rngs=nnx.Rngs(wrapper_rng),
            )

        router_partial = None
        router_checkpoint = None
        if args.stage == "action" and args.condition in ROUTE_CONDITIONS:
            router_dir = args.router_model_dir.expanduser().resolve()
            meta = json.loads((router_dir / "experiment.json").read_text(encoding="utf-8"))
            if meta["stage"] != "router" or meta["condition"] != args.condition:
                raise ValueError("Router checkpoint stage/condition mismatch")
            router_checkpoint = Path(meta["best_router_checkpoint"])
            from openpi.models import model as model_api
            router_partial = model_api.restore_params(router_checkpoint, dtype=jnp.bfloat16)

        resume_partial = None
        resume_checkpoint = None
        start_step = 0
        if args.resume_model_dir is not None:
            resume_dir = args.resume_model_dir.expanduser().resolve()
            resume_meta = json.loads((resume_dir / "experiment.json").read_text(encoding="utf-8"))
            if resume_meta["stage"] != "action" or resume_meta["condition"] != args.condition:
                raise ValueError("Resume checkpoint stage/condition mismatch")
            start_step = int(resume_meta["steps"])
            if args.steps <= start_step:
                raise ValueError(
                    f"Target --steps ({args.steps}) must exceed resume step ({start_step})"
                )
            resume_checkpoint = Path(resume_meta["best_action_checkpoint"])
            from openpi.models import model as model_api
            resume_partial = model_api.restore_params(resume_checkpoint, dtype=jnp.bfloat16)

        checkpoint_steps = {
            int(item.strip())
            for item in args.checkpoint_steps.split(",")
            if item.strip()
        }
        invalid_checkpoint_steps = sorted(
            step for step in checkpoint_steps if step <= start_step or step > args.steps
        )
        if invalid_checkpoint_steps:
            raise ValueError(
                f"Checkpoint steps must be in ({start_step}, {args.steps}]: "
                f"{invalid_checkpoint_steps}"
            )
        if args.lr_schedule == "constant":
            lr_schedule = optax.constant_schedule(args.learning_rate)
        else:
            if args.lr_warmup_steps <= 0:
                raise ValueError("--lr-warmup-steps must be positive")
            if args.lr_decay_start_step is None:
                raise ValueError("--lr-decay-start-step is required")
            if not start_step <= args.lr_decay_start_step < args.steps:
                raise ValueError(
                    "--lr-decay-start-step must be between the start and final steps"
                )
            if args.lr_final is None or not 0.0 < args.lr_final <= args.learning_rate:
                raise ValueError("--lr-final must be in (0, --learning-rate]")

            def lr_schedule(local_count):
                global_step = local_count + start_step
                warmup = args.learning_rate * jnp.minimum(
                    (global_step + 1) / args.lr_warmup_steps, 1.0
                )
                progress = jnp.clip(
                    (global_step - args.lr_decay_start_step)
                    / (args.steps - args.lr_decay_start_step),
                    0.0,
                    1.0,
                )
                cosine = args.lr_final + 0.5 * (
                    args.learning_rate - args.lr_final
                ) * (1.0 + jnp.cos(jnp.pi * progress))
                return jnp.where(
                    global_step < args.lr_warmup_steps,
                    warmup,
                    jnp.where(global_step < args.lr_decay_start_step, args.learning_rate, cosine),
                )

        tx = optax.chain(
            optax.clip_by_global_norm(args.gradient_clip),
            optax.adamw(lr_schedule, weight_decay=args.weight_decay),
        )

        def init_state(rng, partial_base=None, partial_router=None, partial_resume=None):
            model = create_model(rng)
            graph, model_state = nnx.split(model)
            if partial_base is not None:
                model_state.replace_by_pure_dict({"base": partial_base})
            if partial_router is not None:
                model_state.replace_by_pure_dict(partial_router)
            if partial_resume is not None:
                model_state.replace_by_pure_dict(partial_resume)
            model = nnx.merge(graph, model_state)
            params = nnx.state(model)
            frozen_filter = nnx.All(nnx.Param, nnx.Not(train_filter))
            params = nnx_utils.state_map(
                params, frozen_filter,
                lambda parameter: parameter if parameter.value is None else parameter.replace(
                    parameter.value.astype(jnp.bfloat16)
                ),
            )
            return training_utils.TrainState(
                step=jnp.asarray(0, jnp.int32), params=params, model_def=nnx.graphdef(model),
                tx=tx, opt_state=tx.init(params.filter(train_filter)),
                ema_decay=None, ema_params=None,
            )

        init_key = jax.random.key(args.seed)
        state_shape = jax.eval_shape(init_state, init_key)
        state_sharding = sharding.fsdp_sharding(state_shape, mesh, min_size_mbytes=4, log=True)
        base_reference = state_shape.params.to_pure_dict()["base"]
        init_params = args.init_params.expanduser().resolve()
        loaded_base = weight_loaders.CheckpointWeightLoader(str(init_params)).load(base_reference)
        loaded_base = traverse_util.unflatten_dict(
            {
                path: value
                for path, value in traverse_util.flatten_dict(loaded_base).items()
                if not isinstance(value, jax.ShapeDtypeStruct)
            }
        )
        train_state = jax.jit(
            init_state, donate_argnums=(1,),
            in_shardings=(replicated, replicated, replicated, replicated),
            out_shardings=state_sharding,
        )(init_key, loaded_base, router_partial, resume_partial)
        jax.block_until_ready(train_state)
        print(
            f"trainable={parameter_count(state_shape.params.filter(train_filter)):,}/"
            f"{parameter_count(state_shape.params.filter(nnx.Param)):,} devices={jax.device_count()}",
            flush=True,
        )

        metadata = {
            "schema": "pi05_stage_relation_router_v1",
            "completed": False,
            "stage": args.stage,
            "condition": args.condition,
            "config_name": args.config_name,
            "base_model_config": dataclasses.asdict(config.model),
            "implementation_version": "worldla-0.1.0",
            "dataset_repo_id": args.dataset_repo_id,
            "asset_dir": None if args.asset_dir is None else str(args.asset_dir.expanduser().resolve()),
            "asset_id": args.asset_id,
            "route_manifest": str(args.route_manifest.expanduser().resolve()),
            "global_feature_manifest": None if args.global_feature_manifest is None else str(args.global_feature_manifest.expanduser().resolve()),
            "global_feature_dim": None if feature_store is None else feature_store.feature_dim,
            "global_feature_provenance": None if feature_store is None else feature_store.provenance,
            "global_feature_head_deployed": False,
            "trainable_parameter_count": parameter_count(state_shape.params.filter(train_filter)),
            "deployment_parameter_count": parameter_count(state_shape.params.filter(deploy_filter)),
            "training_only_feature_head_parameter_count": parameter_count(state_shape.params.filter(global_head_pattern)),
            "init_params": str(init_params),
            "steps": args.steps,
            "seed": args.seed,
            "router_hidden_dim": args.router_hidden_dim,
            "adapter_dim": args.adapter_dim,
            "num_route_tokens": store.num_route_tokens,
            "num_stages": store.num_stages,
            "prefix_dim": prefix_dim,
            "suffix_dim": suffix_dim,
            "router_active": router_active,
            "message_enabled": message_enabled,
            "position_only": position_only,
            "router_checkpoint_loaded": None if router_checkpoint is None else str(router_checkpoint),
            "resume_checkpoint_loaded": None if resume_checkpoint is None else str(resume_checkpoint),
            "start_step": start_step,
            "optimizer_resumed": False,
            "lr_schedule": args.lr_schedule,
            "lr_peak": args.learning_rate,
            "lr_warmup_steps": args.lr_warmup_steps,
            "lr_decay_start_step": args.lr_decay_start_step,
            "lr_final": args.lr_final,
            "evaluation_checkpoint_steps": sorted(checkpoint_steps),
            "train_data_mode": train_mode,
            "wam_required_at_inference": False,
            "router_frozen_during_action": args.stage == "action" and args.condition in ROUTE_CONDITIONS,
            "stage_shuffle_mode": (
                "per-sample per-step random derangement; validation fixed by global index"
                if args.condition == "stage_randomized" else None
            ),
        }
        teacher_content = {
            "base_lora": "none; matched pi0.5 action-only baseline",
            "router_action_only": "none; random Router updated exclusively by native action loss",
            "global_feature_distill": "single globally mean-pooled Fast-WAM VIDEO hidden vector; no shared-K, teacher action channel, spatial labels or stage labels",
            "static_average": "same-observation shared-K spatial routes averaged over all four stages",
            "stage_randomized": "same-observation shared-K maps; fresh per-sample deranged stage assignment each training step",
            "stage_shuffled": "shared-K routes with one fixed stage permutation (legacy control)",
            "position_only": "observation-independent global training-set four-stage spatial prior",
            "wam_stage_dynamic": "four Fast-WAM shared-K spatial routes; no WAM V or action target",
        }[args.condition]
        metadata["teacher_content"] = teacher_content
        router_source = (
            "none"
            if args.condition == "base_lora"
            else "separate pi0.5 visual and language/state contexts plus stage embedding"
        )
        message_destination = (
            "none"
            if args.condition == "base_lora"
            else "corresponding pi0.5 action suffix stage"
        )
        write_json(output / "run_metadata.json", metadata)

        def train_step(rng, state, observation, actions, route_target):
            model = nnx.merge(state.model_def, state.params)
            model.train()
            def loss_fn(current):
                if args.stage == "router":
                    if args.condition == "global_feature_distill":
                        return current.compute_global_feature_loss(observation, route_target)
                    return current.compute_route_loss(observation, route_target)
                return current.compute_action_loss(rng, observation, actions)
            (loss, metrics), grads = nnx.value_and_grad(
                loss_fn, argnums=nnx.DiffState(0, train_filter), has_aux=True
            )(model)
            trainable = state.params.filter(train_filter)
            updates, opt_state = state.tx.update(grads, state.opt_state, trainable)
            nnx.update(model, optax.apply_updates(trainable, updates))
            return dataclasses.replace(
                state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state
            ), {**metrics, "grad_norm": optax.global_norm(grads)}

        ptrain = jax.jit(
            train_step,
            in_shardings=(replicated, state_sharding, data_sharding, data_sharding, data_sharding),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,),
        )

        def validate_route(state, observation, target):
            model = nnx.merge(state.model_def, state.params); model.eval()
            if args.condition == "global_feature_distill":
                return model.compute_global_feature_loss(observation, target)[1]
            return model.compute_route_loss(observation, target)[1]

        def validate_action(rng, state, observation, actions):
            model = nnx.merge(state.model_def, state.params); model.eval()
            return model.compute_action_loss(rng, observation, actions)[1]

        pvalidate_route = jax.jit(
            validate_route,
            in_shardings=(state_sharding, data_sharding, data_sharding), out_shardings=replicated,
        )
        pvalidate_action = jax.jit(
            validate_action,
            in_shardings=(replicated, state_sharding, data_sharding, data_sharding), out_shardings=replicated,
        )

        checkpointer = ocp.PyTreeCheckpointer()
        iterator = iter(train_loader)
        best_val = float("inf")
        best_step = None
        last_val = None
        started = time.time()
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader); batch = next(iterator)
            if args.stage == "router" and args.condition == "stage_randomized":
                batch["route_target"] = randomize_stage_targets(
                    batch["route_target"], batch["route_global_index"], seed=args.seed, step=step
                )
            observation, actions, route, _, _, _ = split_batch(batch, sharding=data_sharding)
            key = jax.random.fold_in(jax.random.key(args.seed), step - 1)
            with sharding.set_mesh(mesh):
                train_state, metrics = ptrain(key, train_state, observation, actions, route)
            if step % args.log_interval == 0 or step in (1, args.steps):
                current_lr = float(jax.device_get(lr_schedule(step - start_step - 1)))
                row = {
                    "step": step,
                    "learning_rate": current_lr,
                    **{k: float(v) for k, v in jax.device_get(metrics).items()},
                }
                with (output / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"step={step:05d} loss={row['loss']:.6f} "
                    f"grad={row['grad_norm']:.4f} lr={current_lr:.3e}",
                    flush=True,
                )

            if step % args.validation_interval == 0 or step == args.steps:
                totals = {}; count = 0
                for batch_index, val_batch in enumerate(val_loader):
                    if args.max_validation_batches is not None and batch_index >= args.max_validation_batches:
                        break
                    if args.stage == "router" and args.condition == "stage_randomized":
                        val_batch["route_target"] = randomize_stage_targets(
                            val_batch["route_target"], val_batch["route_global_index"], seed=args.seed, step=0
                        )
                    val_obs, val_actions, val_route, val_prior, _, _ = split_batch(
                        val_batch, sharding=data_sharding
                    )
                    with sharding.set_mesh(mesh):
                        if args.stage == "router":
                            values = jax.device_get(pvalidate_route(train_state, val_obs, val_route))
                        else:
                            val_key = jax.random.fold_in(jax.random.key(args.seed + 99991), batch_index)
                            values = jax.device_get(pvalidate_action(val_key, train_state, val_obs, val_actions))
                    batch_count = int(val_route.shape[0]); count += batch_count
                    for name, value in values.items():
                        totals[name] = totals.get(name, 0.0) + float(value) * batch_count
                    if args.stage == "router" and args.condition != "global_feature_distill":
                        target = np.asarray(jax.device_get(val_route), np.float64)
                        prior = np.asarray(jax.device_get(val_prior), np.float64)
                        prior_kl = np.sum(target * (np.log(target + 1e-8) - np.log(prior + 1e-8)), axis=-1).mean()
                        totals["position_prior_kl"] = totals.get("position_prior_kl", 0.0) + float(prior_kl) * batch_count
                val_row = {"step": step, **{k: value / count for k, value in totals.items()}}
                if args.stage == "router" and args.condition != "global_feature_distill":
                    val_row["route_kl_gain_vs_position"] = val_row["position_prior_kl"] - val_row["route_kl"]
                with (output / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(val_row, sort_keys=True) + "\n")
                selection = (
                    val_row["feature_loss"] if feature_store is not None
                    else val_row["route_kl"] if args.stage == "router"
                    else val_row["action_loss"]
                )
                if feature_store is not None:
                    print(f"validation step={step:05d} feature_loss={selection:.6f} cos={val_row['feature_cosine']:.4f}", flush=True)
                elif args.stage == "router":
                    print(
                        f"validation step={step:05d} kl={selection:.6f} "
                        f"cos={val_row['route_cosine']:.4f} "
                        f"gain_vs_position={val_row['route_kl_gain_vs_position']:.6f}", flush=True
                    )
                else:
                    print(
                        f"validation step={step:05d} action_loss={selection:.6f} "
                        f"gate={val_row['adapter_gate']:.4f}", flush=True
                    )
                if not np.isfinite(selection):
                    raise RuntimeError(
                        f"Non-finite validation objective at step {step}: {selection}"
                    )
                if selection < best_val:
                    best_val, best_step = selection, step
                last_val = selection

            if step in checkpoint_steps and step < args.steps:
                snapshot_dir = output / "eval_snapshots" / f"step_{step:06d}"
                snapshot_checkpoint = (
                    snapshot_dir / "action_checkpoints" / f"final_step_{step:06d}"
                )
                snapshot_model = nnx.merge(train_state.model_def, train_state.params)
                snapshot_filter = nnx.All(
                    nnx.Param, nnx.Any(lora_pattern, router_pattern, adapter_pattern)
                )
                checkpointer.save(
                    snapshot_checkpoint,
                    {"params": nnx.state(snapshot_model, snapshot_filter)},
                )
                snapshot_experiment = {
                    **metadata,
                    "completed": True,
                    "steps": step,
                    "best_router_checkpoint": None,
                    "best_val_route_kl": None,
                    "best_action_checkpoint": str(snapshot_checkpoint),
                    "best_val_action_loss": last_val,
                    "best_observed_validation_value": best_val,
                    "best_observed_validation_step": best_step,
                    "checkpoint_policy": "continuous_training_evaluation_snapshot",
                    "adapter_params": str(snapshot_checkpoint),
                    "final_adapter_params": str(snapshot_checkpoint),
                    "router_frozen_during_action": args.condition in ROUTE_CONDITIONS,
                    "teacher_content": teacher_content,
                    "router_source": router_source,
                    "message_destination": message_destination,
                    "action_objective": "native pi0.5 flow-matching loss only",
                    "batch_size": args.batch_size,
                    "fsdp_devices": args.fsdp_devices,
                    "learning_rate": args.learning_rate,
                    "elapsed_seconds": time.time() - started,
                }
                write_json(snapshot_dir / "experiment.json", snapshot_experiment)
                print(f"Saved evaluation snapshot: {snapshot_dir}", flush=True)

        if best_step is None or last_val is None:
            raise RuntimeError("No validation checkpoint produced")
        # Validation remains periodic, but Orbax writes exactly once at the end.
        # This uses a fixed training budget for every control and prevents
        # checkpoint accumulation from exhausting /data1.
        group = "router_checkpoints" if args.stage == "router" else "action_checkpoints"
        final_checkpoint = output / group / f"final_step_{args.steps:05d}"
        model = nnx.merge(train_state.model_def, train_state.params)
        save_filter = (
            nnx.All(nnx.Param, router_pattern)
            if args.stage == "router" else deploy_filter
        )
        checkpointer.save(
            final_checkpoint, {"params": nnx.state(model, save_filter)}
        )
        if args.stage == "action":
            write_json(output / "latest_action_checkpoint.json", {
                "step": args.steps,
                "checkpoint": str(final_checkpoint),
                "validation_action_loss": last_val,
            })
        final_adapter = final_checkpoint if args.stage == "action" else None
        experiment = {
            **metadata,
            "completed": True,
            "best_router_checkpoint": str(final_checkpoint) if args.stage == "router" else None,
            "best_val_route_kl": last_val if args.stage == "router" and feature_store is None else None,
            "best_val_feature_loss": last_val if feature_store is not None else None,
            "best_action_checkpoint": str(final_checkpoint) if args.stage == "action" else None,
            "best_val_action_loss": last_val if args.stage == "action" else None,
            "best_observed_validation_value": best_val,
            "best_observed_validation_step": best_step,
            "checkpoint_policy": (
                "final_step_plus_evaluation_snapshots"
                if checkpoint_steps else "final_step_only"
            ),
            "adapter_params": str(final_checkpoint) if args.stage == "action" else None,
            "final_adapter_params": str(final_adapter) if final_adapter is not None else None,
            "router_frozen_during_action": args.stage == "action" and args.condition in ROUTE_CONDITIONS,
            "teacher_content": teacher_content,
            "router_source": router_source,
            "message_destination": message_destination,
            "action_objective": "native pi0.5 flow-matching loss only",
            "batch_size": args.batch_size,
            "fsdp_devices": args.fsdp_devices,
            "learning_rate": args.learning_rate,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output / "experiment.json", experiment)
        print(f"Saved {args.stage} result to {output}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
