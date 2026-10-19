"""Child-process shim for evaluators importing their policy by module name."""
import runpy
import sys
from pathlib import Path


def main():
    from worldla.integrations import robotwin
    script = sys.argv[1]
    sys.path.insert(0, str(Path(script).absolute().parent))
    root = Path(script).absolute().parent.parent
    planner = root / 'envs/curobo/src'
    if planner.is_dir():
        sys.path.insert(0, str(planner))
    from worldla.integrations.rendering import bind_visible_renderer
    bind_visible_renderer()
    sys.modules['worldla_policy'] = robotwin
    sys.argv = [script,*sys.argv[2:]]
    runpy.run_path(script,run_name='__main__')


if __name__=='__main__':
    main()
