#!/usr/bin/env python3
"""Exercise installed WorldLA's two training stages on real cached/local data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def verify(output, steps):
    # Validation of saved parameters is CPU-only; training children have exited.
    os.environ['JAX_PLATFORMS']='cpu'
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict
    from worldla.models.interfaces import restore_params

    result={}
    for stage in ('router','action'):
        folder=output/stage
        meta=json.loads((folder/'experiment.json').read_text())
        rows=[json.loads(line) for line in (folder/'train_metrics.jsonl').read_text().splitlines()]
        validation=[json.loads(line) for line in (folder/'validation_metrics.jsonl').read_text().splitlines()]
        assert meta['completed'] and meta['steps']==steps and meta['stage']==stage
        assert [row['step'] for row in rows]==list(range(1,steps+1))
        assert all(np.isfinite(row['loss']) and np.isfinite(row['grad_norm'])
                   and row['grad_norm']>0 for row in rows)
        assert validation[-1]['step']==steps
        result[stage]={'experiment':str(folder/'experiment.json'),
                       'train_losses':[row['loss'] for row in rows],
                       'gradient_norms':[row['grad_norm'] for row in rows],
                       'validation':validation[-1]}
    router_meta=json.loads((output/'router/experiment.json').read_text())
    action_meta=json.loads((output/'action/experiment.json').read_text())
    assert action_meta['router_frozen_during_action'] is True
    assert action_meta['router_checkpoint_loaded']==router_meta['best_router_checkpoint']
    snapshot=json.loads((output/'action/eval_snapshots/step_000002/experiment.json').read_text())
    router=flatten_dict(restore_params(router_meta['best_router_checkpoint'],dtype=None))
    before=flatten_dict(restore_params(snapshot['best_action_checkpoint'],dtype=None))
    after=flatten_dict(restore_params(action_meta['best_action_checkpoint'],dtype=None))
    assert before.keys()==after.keys()
    counts={'changed_lora_leaves':0,'changed_adapter_leaves':0,'frozen_router_leaves':0}
    for key,value in after.items():
        old=before[key]
        if value is None:
            assert old is None
            continue
        value=np.asarray(value)
        assert np.isfinite(value.astype(np.float32)).all()
        changed=not np.array_equal(np.asarray(old),value)
        if key[0]=='stage_relation_router':
            assert not changed, f'Frozen router changed: {key}'
            expected=np.asarray(jnp.asarray(router[key],dtype=jnp.bfloat16))
            assert np.array_equal(expected,value),f'Pretrained router was not restored: {key}'
            counts['frozen_router_leaves']+=1
        elif key[0]=='stage_message_adapter':
            counts['changed_adapter_leaves']+=int(changed)
        elif key[0]=='base':
            assert any('lora' in str(part) for part in key),f'Unexpected dense-base parameter: {key}'
            counts['changed_lora_leaves']+=int(changed)
        else:
            raise AssertionError(f'Unexpected checkpoint parameter: {key}')
    assert all(value>0 for value in counts.values()),counts
    return {**result,'parameter_checks':counts,'full_training_convergence_validated':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('lerobot-root','asset-dir','route-manifest','init-params','tokenizer-path','output'):
        parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--dataset-repo-id',required=True)
    parser.add_argument('--asset-id',required=True)
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=4)
    args=parser.parse_args()
    gpu_ids=args.gpus.split(',')
    if not all(x.isdigit() for x in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):
        raise ValueError('Provide distinct numeric GPU IDs')
    if args.steps<3 or args.batch_size<=0 or args.batch_size%len(gpu_ids):
        raise ValueError('Need steps>=3 and a positive global batch divisible by the GPU count')
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    for gpu in gpu_ids:
        memory=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits'],text=True)
        if int(memory.strip())>=512:
            raise RuntimeError(f'GPU {gpu} is occupied; refusing to launch training')
    output.mkdir(parents=True)
    cache=output/'runtime_cache'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpus,'JAX_PLATFORMS':'cuda',
         'JAX_DEFAULT_MATMUL_PRECISION':'highest','XLA_PYTHON_CLIENT_PREALLOCATE':'false',
         'XLA_PYTHON_CLIENT_MEM_FRACTION':'.80','PYTHONUNBUFFERED':'1',
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1',
         'WANDB_MODE':'disabled','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'HF_HOME':str(cache/'hf_home'),'HF_DATASETS_CACHE':str(cache/'hf_datasets'),
         'WORLDLA_DATA_HOME':str(cache/'worldla'),'JAX_COMPILATION_CACHE_DIR':str(cache/'jax')}
    env.pop('LD_PRELOAD',None)
    import worldla.training.runner
    import worldla
    state={'training_module':worldla.training.runner.__file__,'backend_module':worldla.__file__,
           'gpus':gpu_ids,'steps_per_stage':args.steps,'batch_size':args.batch_size,
           'runs':[],'state':'running'}
    (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2),flush=True)
    common=[sys.executable,'-m','worldla','train','--condition','wam_stage_dynamic',
        '--lerobot-root',str(args.lerobot_root.resolve()),'--dataset-repo-id',args.dataset_repo_id,
        '--asset-dir',str(args.asset_dir.resolve()),'--asset-id',args.asset_id,
        '--route-manifest',str(args.route_manifest.resolve()),'--init-params',str(args.init_params.resolve()),
        '--tokenizer-path',str(args.tokenizer_path.resolve()),'--runtime-cache-dir',str(cache),
        '--steps',str(args.steps),'--batch-size',str(args.batch_size),'--fsdp-devices',str(len(gpu_ids)),
        '--validation-interval',str(args.steps),'--max-validation-batches','1','--log-interval','1']
    try:
        for stage in ('router','action'):
            command=[*common,'--stage',stage,'--output-dir',str(output/stage)]
            if stage=='action':
                command+=['--router-model-dir',str(output/'router'),'--checkpoint-steps','2',
                          '--lr-schedule','warmup_plateau_cosine','--lr-warmup-steps','1',
                          '--lr-decay-start-step','2','--lr-final','2e-5']
            print(f'Starting {stage}: GPUs={args.gpus}, steps={args.steps}',flush=True)
            with (output/f'{stage}.log').open('w') as log:
                run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['runs'].append({'stage':stage,'returncode':run.returncode,'command':command})
            (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            if run.returncode:
                raise RuntimeError(f'{stage} failed; inspect {output/(stage+".log")}')
        state['verification']=verify(output,args.steps)
        state['state']='complete'
    except BaseException as exc:
        state['state']='failed'
        state['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':
    main()
