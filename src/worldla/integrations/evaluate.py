"""Launch a user-installed RoboTwin evaluator without copying simulator assets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root',type=Path,required=True)
    parser.add_argument('--eval-script',type=Path,default=Path('script/eval_policy.py'))
    parser.add_argument('--deploy-config',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--task-config',choices=('demo_clean','demo_randomized'),default='demo_clean')
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed-group',type=int,default=0)
    parser.add_argument('--inference-seed',type=int,default=20260901)
    parser.add_argument('--old-plan-weight',type=float,default=.75)
    parser.add_argument('--accepted-seed-dir',type=Path)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--run-tag',default='worldla_full')
    parser.add_argument('--dry-run',action='store_true')
    args,extra = parser.parse_known_args()
    root = args.robotwin_root.expanduser().resolve()
    script = args.eval_script if args.eval_script.is_absolute() else root/args.eval_script
    if not script.is_file() or not args.deploy_config.is_file():
        raise FileNotFoundError('Provide an installed RoboTwin evaluator and its matching deploy YAML')
    if args.episodes <= 0 or not 0 <= args.old_plan_weight <= 1:
        raise ValueError('Invalid episode count or old-plan weight')
    instruction = 'seen' if args.task_config=='demo_clean' else 'unseen'
    overrides = ['--policy_name','worldla_policy','--model_dir',str(args.model_dir.resolve()),
        '--task_name',args.task,'--task_config',args.task_config,'--instruction_type',instruction,
        '--eval_episode_num',str(args.episodes),'--seed',str(args.seed_group),
        '--inference_seed',str(args.inference_seed),'--old_plan_weight',str(args.old_plan_weight),
        '--ckpt_setting',args.run_tag,'--train_config_name','worldla_pi05','--model_name','worldla',
        '--pi0_step','10','--record_policy_debug','false',
        '--eval_video_log','false']
    for name,value in [('accepted_seed_dir',args.accepted_seed_dir),('eval_summary_path',args.summary)]:
        if value is not None:
            overrides += ['--'+name,str(value.resolve())]
    # Keep compatibility-root symlinks intact: the evaluator resolves its own
    # simulator/assets root from __file__, not from the symlink's target.
    command = [sys.executable,'-m','worldla.integrations.robotwin_runner',str(script.absolute()),
               '--config',str(args.deploy_config.resolve()),'--overrides',*overrides,*extra]
    if args.dry_run:
        print(json.dumps({'cwd':str(root),'command':command},indent=2))
        return
    subprocess.run(command,cwd=root,check=True)
