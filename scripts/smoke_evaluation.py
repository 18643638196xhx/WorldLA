#!/usr/bin/env python3
"""Run isolated, fixed-seed closed-loop smoke tests using the installed WorldLA package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root',type=Path,required=True)
    parser.add_argument('--eval-script',type=Path,required=True)
    parser.add_argument('--deploy-config',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--merged',type=Path)
    parser.add_argument('--seed-cache',type=Path,required=True)
    parser.add_argument('--task',default='press_stapler')
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.robotwin_root.resolve()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite smoke run: {output}')
    payload=json.loads(args.seed_cache.read_text())
    if payload['task_name']!=args.task or payload['task_config']!='demo_clean':
        raise ValueError('Fixed seed cache does not match the smoke task')
    if len(payload['episodes'])<args.episodes or args.episodes<=0:
        raise ValueError('Need sufficient fixed accepted seeds')
    output.mkdir(parents=True)
    runtime=output/'runtime'
    runtime.mkdir()
    for name,target in {
        'envs':root/'envs','assets':root/'assets','description':root/'description',
        'task_config':root/'env_cfg/task_config','scripts':root/'scripts',
    }.items():
        if not target.is_dir():
            raise FileNotFoundError(target)
        (runtime/name).symlink_to(target,target_is_directory=True)
    (runtime/'env_cfg').mkdir()
    (runtime/'env_cfg/task_config').symlink_to(root/'env_cfg/task_config',target_is_directory=True)
    (runtime/'script').mkdir()
    (runtime/'script/eval_policy.py').symlink_to(args.eval_script.resolve())
    for name in ('generate_episode_instructions.py','test_render.py'):
        sibling=args.eval_script.resolve().parent/name
        if sibling.exists():
            (runtime/'script'/name).symlink_to(sibling)
    seed_dir=output/'accepted_seeds'
    seed_dir.mkdir()
    small={**payload,'episodes':payload['episodes'][:args.episodes]}
    seed_file=seed_dir/args.seed_cache.name
    seed_file.write_text(json.dumps(small,indent=2)+'\n')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION='.8',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    env.pop('LD_PRELOAD',None)
    records=[]
    for variant,model in [('original',args.original),('merged',args.merged)]:
        if model is None:
            continue
        memory=subprocess.run(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        if int(memory.stdout.strip())>=512:
            raise RuntimeError(f'GPU {args.gpu} is no longer idle; refusing to start {variant}')
        summary=output/f'{variant}.summary.jsonl'
        command=[sys.executable,'-m','worldla','eval-robotwin','--robotwin-root',str(runtime),
                 '--deploy-config',str(args.deploy_config.resolve()),'--model-dir',str(model.resolve()),
                 '--task',args.task,'--episodes',str(args.episodes),'--accepted-seed-dir',str(seed_dir),
                 '--run-tag',f'worldla_package_smoke_{variant}','--summary',str(summary)]
        print(f'Starting {variant}: GPU {args.gpu}, {args.task}, fixed seeds {[x["seed"] for x in small["episodes"]]}',flush=True)
        with (output/f'{variant}.log').open('w') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record={'variant':variant,'returncode':result.returncode,'summary':str(summary),
                'log':str(output/f'{variant}.log')}
        if result.returncode==0:
            summaries=[json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
            if len(summaries)!=1 or summaries[0].get('trials')!=args.episodes:
                raise RuntimeError(f'{variant}: expected one complete {args.episodes}-episode summary')
            record['result']=summaries[0]
        records.append(record)
        (output/'status.json').write_text(json.dumps({'runs':records,'formal_benchmark':False},indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{variant} smoke evaluation failed; inspect {output/ (variant+".log")}')
        print(f'{variant} smoke evaluation finished; summary={summary}',flush=True)
    print(json.dumps({'runs':records,'formal_benchmark':False},indent=2),flush=True)


if __name__=='__main__':
    main()
