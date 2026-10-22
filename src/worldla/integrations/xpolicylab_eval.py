"""Launch the native RoboTwin XPolicyLab evaluator in its own Python environment."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--env-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--task', required=True)
    parser.add_argument('--task-config', choices=('demo_clean', 'demo_randomized'), default='demo_clean')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-group', type=int, default=0)
    parser.add_argument('--run-tag', default='worldla_full')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.episodes <= 0 or args.seed_group < 0:
        parser.error('Invalid port, episode count or seed group')
    if not args.host or not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_tag):
        parser.error('Provide a host and an alphanumeric/dot/dash/underscore run tag')
    root = args.robotwin_root.expanduser().resolve()
    script = root / 'scripts/eval_policy_xpolicylab.py'
    env_python = args.env_python.expanduser().absolute()
    for path in (script, env_python, root / 'env_cfg/task_config' / (args.task_config + '.yml')):
        if not path.is_file():
            raise FileNotFoundError(path)
    runner = Path(__file__).with_name('xpolicylab_env_runner.py')
    command = [str(env_python), str(runner), str(script), '--bench_name', 'RoboTwin',
               '--task_name', args.task, '--policy_name', 'WorldLA',
               '--host', args.host, '--port', str(args.port), '--protocol', 'ws',
               '--eval_batch', 'false', '--root_dir', str(root), '--device_id', '0',
               '--seed', str(args.seed_group), '--task_config', args.task_config,
               '--test_num', str(args.episodes), '--instruction_type',
               'seen' if args.task_config == 'demo_clean' else 'unseen',
               '--additional_info', f'action_type=joint,ckpt_setting={args.run_tag}']
    if args.dry_run:
        print(json.dumps({'cwd': str(root), 'command': command,
                          'output_root': str(root / 'eval_result'),
                          'starts_policy_server': False}, indent=2))
        return
    subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=True)
