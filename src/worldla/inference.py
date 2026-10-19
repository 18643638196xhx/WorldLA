"""Run one inference without issuing simulator/robot commands."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='NPZ: state, cam_high, cam_left_wrist, cam_right_wrist; RGB uint8 CHW')
    parser.add_argument('--dummy', action='store_true', help='Explicit shape smoke test; NEVER execute outputs on a robot')
    parser.add_argument('--prompt', default='Place the can in the basket.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20260901)
    args = parser.parse_args()
    if bool(args.input) == bool(args.dummy):
        parser.error('Choose exactly one of --input or --dummy')
    if args.output is not None and args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    cameras = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    if args.dummy:
        observation = {'state': np.zeros(14, np.float32),
                       'images': {name: np.zeros((3,224,224), np.uint8) for name in cameras}}
    else:
        with np.load(args.input, allow_pickle=False) as data:
            observation = {'state': data['state'], 'images': {name: data[name] for name in cameras}}
    observation['prompt'] = args.prompt
    from worldla.policy import WorldLAPolicy
    policy = WorldLAPolicy(args.model_dir, seed=args.seed)
    actions = policy.infer(observation)['actions']
    if actions.shape != (50,14) or not np.isfinite(actions).all():
        raise RuntimeError(f'Invalid action output: {actions.shape}')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, actions=actions)
    print(f'Finite action chunk: {actions.shape}; no robot commands were executed.')
