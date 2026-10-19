"""Process-local caches, with no symlinks to a development checkout."""
import os
from pathlib import Path
import shutil


def configure_runtime(cache_dir=None, tokenizer_path=None):
    root = Path(cache_dir or os.environ.get('WORLDLA_CACHE_DIR', Path.home() / '.cache/worldla')).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['OPENPI_RUNTIME_CACHE_DIR'] = str(root)
    os.environ.setdefault('OPENPI_DATA_HOME', str(root / 'openpi'))
    if tokenizer_path is not None:
        source = Path(tokenizer_path).expanduser().resolve()
        target = Path(os.environ['OPENPI_DATA_HOME']) / 'big_vision/paligemma_tokenizer.model'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('A different tokenizer already exists in the selected cache')
        if not target.exists():
            shutil.copyfile(source, target)
    return root
