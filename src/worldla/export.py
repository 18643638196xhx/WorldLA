"""CPU-only export of trained base+increments; source checkpoints are read-only."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from worldla.checkpoints import assemble_original, fold_lora, validate_tree
from worldla.io import write_json


def hashes(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=('original','merged'), default='original')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--base-params', type=Path, help='Override metadata path after moving a training run')
    parser.add_argument('--action-params', type=Path)
    parser.add_argument('--assets-dir', type=Path)
    args = parser.parse_args()
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.models import gemma, model as model_api
    from openpi.models.pi0_config import Pi0Config
    from worldla.model import StageRelationRouterPi
    source, target = args.model_dir.resolve(), args.output_dir.resolve()
    meta = json.loads((source/'experiment.json').read_text())
    if not meta.get('completed') or meta['stage'] != 'action':
        raise ValueError('Export requires a completed action-training run')
    base = Path(args.base_params or meta['init_params']).resolve()
    delta = Path(args.action_params or meta['adapter_params']).resolve()
    for directory in (source,base,delta):
        if target == directory or directory in target.parents or target in directory.parents:
            raise ValueError('Export directory overlaps a source checkpoint')
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite: {target}')
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)
    cfg = meta.get('base_model_config')
    if cfg is None and meta['config_name'] not in {'multi_task_4','worldla_pi05'}:
        raise ValueError('Legacy run has no explicit model configuration')
    model_cfg = Pi0Config(**cfg) if cfg else Pi0Config(
        pi05=True, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    kwargs = {key: meta[key] for key in ('prefix_dim','suffix_dim','num_route_tokens','num_stages',
              'router_hidden_dim','adapter_dim','router_active','message_enabled','position_only')}
    def expected(config):
        model = nnx.eval_shape(lambda: StageRelationRouterPi(config.create(jax.random.key(0)),
                                **kwargs, rngs=nnx.Rngs(0)))
        return traverse_util.flatten_dict(nnx.state(model).to_pure_dict())
    print('Hashing original checkpoints', flush=True)
    before = {'base': hashes(base), 'increments': hashes(delta)}
    base_values = model_api.restore_params(base, restore_type=np.ndarray, dtype=jnp.bfloat16)
    delta_values = model_api.restore_params(delta, restore_type=np.ndarray, dtype=jnp.bfloat16)
    flat_base = {('base',)+key: value for key,value in traverse_util.flatten_dict(base_values).items()}
    flat_delta = traverse_util.flatten_dict(delta_values)
    full = assemble_original(flat_base,flat_delta)
    validate_tree(expected(model_cfg),full)
    operations = []
    merged = args.variant == 'merged'
    if merged:
        scales = {name: gemma.get_config(variant).lora_configs['attn'].scaling_value for name,variant in
                  [('vision_language',model_cfg.paligemma_variant),('action',model_cfg.action_expert_variant)]}
        full,operations = fold_lora(flat_base,flat_delta,scales)
        model_cfg = dataclasses.replace(model_cfg,
            paligemma_variant=model_cfg.paligemma_variant.removesuffix('_lora'),
            action_expert_variant=model_cfg.action_expert_variant.removesuffix('_lora'))
    validate_tree(expected(model_cfg),full)
    target.parent.mkdir(parents=True,exist_ok=True)
    staging = target.parent/f'.{target.name}.exporting.{os.getpid()}'
    staging.mkdir()
    with ocp.PyTreeCheckpointer() as cp:
        cp.save(staging/'params', {'params': traverse_util.unflatten_dict(full)})
    restored = traverse_util.flatten_dict(model_api.restore_params(staging/'params',restore_type=np.ndarray))
    validate_tree(expected(model_cfg),restored)
    for key,value in full.items():
        if value is None:
            equal = restored[key] is None
        else:
            equal = value.dtype == restored[key].dtype and np.array_equal(value,restored[key])
        if not equal:
            raise ValueError(f'Export round-trip mismatch: {key}')
    if before != {'base': hashes(base), 'increments': hashes(delta)}:
        raise RuntimeError('Source parameters changed during export')
    asset = meta['asset_id']
    (staging/'assets'/asset).mkdir(parents=True)
    shutil.copyfile(Path(args.assets_dir or meta['asset_dir'])/asset/'norm_stats.json',
                    staging/'assets'/asset/'norm_stats.json')
    shutil.copyfile(args.tokenizer,staging/'assets/paligemma_tokenizer.model')
    write_json(staging/'config.json', {
        'schema': 'worldla_full_folded_v1' if merged else 'worldla_full_original_v1',
        'lora_merged': merged, 'architecture': 'StageRelationRouterPi',
        'base_model_config': dataclasses.asdict(model_cfg),'router_config':kwargs,'asset_id':asset,
        'source_step':meta['steps'],'source_lr_schedule':meta['lr_schedule'],
        'requires_base_checkpoint':False,'requires_fastwam':False,
        'deployment': {'denoising_steps':10,'execute_k':10,'old_plan_weight':.75,'new_plan_weight':.25,
                       'gripper_mode':'switch','gripper_new_threshold':.5},
    })
    write_json(staging/'export_report.json', {
        'source_files_unchanged':True,'round_trip_all_values_equal':True,
        'source_fingerprints':before,'parameter_leaves':len(full),'lora_pairs_merged':operations,
        'numerical_action_validation':'not_run','task_sr_re_evaluated':False,
        'warning':'Parameter equality is not rollout validation; folding may change low-precision results.'})
    write_json(staging/'files_sha256.json',hashes(staging))
    staging.rename(target)
    print(f'Exported {args.variant}: {target}; original checkpoints unchanged.')
