import os
import pathlib


def configure_runtime_dirs(default_root: pathlib.Path | None = None, *, log_prefix: str = "[runtime]") -> pathlib.Path:
    """Route temp and cache directories into the workspace by default.

    This avoids filling the system partition when Hugging Face `datasets`,
    PyArrow, Transformers, or JAX materialize temporary/cache files.
    Existing environment variables still take precedence.
    """

    if default_root is None:
        default_root = pathlib.Path.home() / ".cache" / "worldla"

    cache_root = pathlib.Path(os.environ.get("OPENPI_RUNTIME_CACHE_DIR", default_root)).resolve()
    tmp_dir = cache_root / "tmp"
    hf_home = cache_root / "hf_home"
    hf_datasets = cache_root / "hf_datasets"
    hf_hub = cache_root / "hf_hub"
    openpi_home = cache_root / "openpi"
    jax_cache = cache_root / "jax"

    for path in (tmp_dir, hf_home, hf_datasets, hf_hub, openpi_home, jax_cache):
        path.mkdir(parents=True, exist_ok=True)

    # WorldLA portability change: preserve caller caches and credentials.
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_hub))
    os.environ.setdefault("OPENPI_DATA_HOME", str(openpi_home))

    print(
        f"{log_prefix} runtime cache configured:",
        f"TMPDIR={os.environ['TMPDIR']}",
        f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}",
        f"HF_HOME={os.environ['HF_HOME']}",
        f"OPENPI_DATA_HOME={os.environ['OPENPI_DATA_HOME']}",
        f"JAX_COMPILATION_CACHE_DIR={jax_cache}",
    )
    return cache_root
