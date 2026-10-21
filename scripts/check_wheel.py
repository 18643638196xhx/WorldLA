#!/usr/bin/env python3
"""Ensure every source module is present in a distributable wheel."""
import argparse
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel',type=Path)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]/'src'
    expected={str(path.relative_to(source)) for path in source.rglob('*.py')}
    with zipfile.ZipFile(args.wheel) as archive:
        names=set(archive.namelist())
    missing=expected-names
    if missing:
        raise RuntimeError(f'Modules missing from wheel: {sorted(missing)}')
    unexpected={name for name in names if name.startswith(('openpi/','openpi_client/'))}
    if unexpected:
        raise RuntimeError(f'Legacy namespace leaked into wheel: {sorted(unexpected)}')
    print(f'All {len(expected)} source modules are present in the wheel.')


if __name__=='__main__':
    main()
