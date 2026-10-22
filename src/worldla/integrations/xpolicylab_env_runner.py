"""Standalone env-Python shim: renderer binding without model/JAX dependencies."""
from pathlib import Path
import runpy
import sys


def main():
    script = Path(sys.argv[1]).absolute()
    root = script.parent.parent
    # Running this file by absolute path works even if WorldLA isn't pip-
    # installed in the simulator environment. __init__ is deliberately lazy.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    for path in (root, script.parent, root / 'envs/curobo/src'):
        if path.is_dir():
            sys.path.insert(0, str(path))
    from worldla.integrations.rendering import bind_visible_renderer
    bind_visible_renderer()
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
