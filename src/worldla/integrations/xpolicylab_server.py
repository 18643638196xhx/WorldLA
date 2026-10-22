"""Serve WorldLA using XPolicyLab's official websocket implementation."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import sys


def load_server_api(root):
    root = Path(root).expanduser().resolve()
    for name in ('model_template.py', 'client_server/ws/model_server.py'):
        if not (root / name).is_file():
            raise FileNotFoundError(f'Not an XPolicyLab checkout: {root / name}')
    # Official transport uses both XPolicyLab.* and client_server.* imports.
    for prefix in ('XPolicyLab', 'client_server'):
        loaded = sys.modules.get(prefix)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f'{prefix} was already imported from a different checkout')
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module('client_server.ws.model_server')
    except ImportError as exc:
        raise ImportError('Install requirements-xpolicylab.txt in a NEW policy environment; '
                          'use the documented XPolicyLab revision') from exc
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('XPolicyLab server resolved to a different checkout')
    return module.PolicyServer, module.PolicyServerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xpolicylab-root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--inference-seed', type=int, default=20260901)
    parser.add_argument('--old-plan-weight', type=float)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.host or not 1 <= args.port <= 65535:
        parser.error('Provide a nonempty host and a port in [1,65535]')
    if args.old_plan_weight is not None and not 0 <= args.old_plan_weight <= 1:
        parser.error('old-plan-weight must lie in [0,1]')
    if not 0 <= args.inference_seed < 2**32:
        parser.error('inference-seed must fit uint32')
    root, model_dir = args.xpolicylab_root.expanduser().resolve(), args.model_dir.expanduser().resolve()
    if not (root / 'client_server/ws/model_server.py').is_file():
        raise FileNotFoundError(f'Missing official XPolicyLab websocket server: {root}')
    from worldla.policy import validate_checkpoint_config
    validate_checkpoint_config(json.loads((model_dir / 'config.json').read_text()))
    if args.dry_run:
        print(json.dumps({'transport': 'XPolicyLab official websocket', 'xpolicylab_root': str(root),
                          'model_dir': str(model_dir), 'host': args.host, 'port': args.port,
                          'model_loaded': False, 'server_started': False}, indent=2))
        return
    from worldla.integrations.xpolicylab import Model
    model = Model({'model_dir': str(model_dir), 'inference_seed': args.inference_seed,
                   'old_plan_weight': args.old_plan_weight, 'action_type': 'joint'})
    # Initialize the CUDA policy before the optional transport's OpenCV/HDF5
    # dependencies; reversing this order crashes native initialization on the
    # tested robotics runtime. The wire protocol/model equations are unchanged.
    server_class, config_class = load_server_api(root)
    server = server_class(model, config_class(host=args.host, port=args.port))
    print(f'[worldla/xpolicylab] Starting official transport at ws://{args.host}:{args.port}; '
          'one evaluation stream per server', flush=True)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print('[worldla/xpolicylab] Server stopped.', flush=True)
