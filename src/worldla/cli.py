"""Lazy command dispatch; --help works without loading model weights."""
import argparse
import importlib
import os
from pathlib import Path
import sys

COMMANDS = {
    'infer': 'worldla.inference',
    'train': 'worldla.training.runner',
    'build-routes': 'worldla.data.build_routes',
    'build-anchors': 'worldla.data.build_anchors',
    'export': 'worldla.export',
    'eval-robotwin': 'worldla.integrations.evaluate',
    'serve-xpolicylab': 'worldla.integrations.xpolicylab_server',
    'eval-xpolicylab': 'worldla.integrations.xpolicylab_eval',
}


def load_yaml_arguments(path):
    import yaml
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError('Config must contain a mapping of CLI option names to values')
    result = []
    for key, value in payload.items():
        if value is None or value is False:
            continue
        result.append('--' + str(key).replace('_', '-'))
        if value is not True:
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = os.path.expandvars(str(item))
                if '${' in text:
                    raise ValueError(f'Unresolved environment variable in option {key}')
                result.append(text)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description='WorldLA model, two-stage training and deployment')
    parser.add_argument('command', choices=COMMANDS)
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return
    command = argv.pop(0)
    if command not in COMMANDS:
        parser.error(f'unknown command: {command}')
    if '--config' in argv:
        index = argv.index('--config')
        if index + 1 >= len(argv):
            parser.error('--config requires a YAML file')
        arguments = load_yaml_arguments(argv[index + 1])
        del argv[index:index + 2]
        argv = arguments + argv
    old = sys.argv
    try:
        sys.argv = ['worldla ' + command, *argv]
        importlib.import_module(COMMANDS[command]).main()
    finally:
        sys.argv = old
