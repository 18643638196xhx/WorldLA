import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from worldla.cli import load_yaml_arguments, main
from worldla.replanning import blend_plan
from worldla.checkpoints import assemble_original, validate_tree


def test_package_help_is_lazy(capsys):
    main(['--help'])
    assert 'build-routes' in capsys.readouterr().out


def test_yaml_flags_are_not_shell_commands(tmp_path,monkeypatch):
    path=tmp_path/'config.yaml'
    path.write_text('steps: 10\nflag: true\nskip: false\noutput_dir: ${TEST_WORLDLA_OUTPUT}\n')
    monkeypatch.setenv('TEST_WORLDLA_OUTPUT','a path with spaces')
    assert load_yaml_arguments(path)==['--steps','10','--flag','--output-dir','a path with spaces']
    monkeypatch.delenv('TEST_WORLDLA_OUTPUT')
    with pytest.raises(ValueError,match='Unresolved'):
        load_yaml_arguments(path)


@pytest.mark.parametrize('weight',[0,.25,.5,.75,1])
def test_plan_alignment_and_gripper_switch(weight):
    old=np.arange(700,dtype=float).reshape(50,14)
    new=np.zeros_like(old)
    result=blend_plan(old,new,old_weight=weight)
    arms=[i for i in range(14) if i not in [6,13]]
    np.testing.assert_array_equal(result[:10,arms],weight*old[10:20,arms])
    np.testing.assert_array_equal(result[:10,[6,13]],old[10:20,[6,13]] if weight>.5 else 0)
    np.testing.assert_array_equal(result[10:],new[10:])
    np.testing.assert_array_equal(blend_plan(None,new,old_weight=weight),new)


def test_assemble_preserves_lora_and_detects_collisions():
    base={('base','w'):np.ones((2,2))}
    delta={('base','lora_a'):np.ones((2,1)),('adapter','bias'):None}
    result=assemble_original(base,delta)
    assert result[('base','w')] is base[('base','w')]
    assert result[('adapter','bias')] is None
    with pytest.raises(ValueError,match='collision'):
        assemble_original(base,base)
    with pytest.raises(ValueError,match='keys differ'):
        validate_tree(base,{})


def test_training_config_has_same_main_architecture():
    from worldla.config import create_training_config
    cfg=create_training_config(SimpleNamespace(asset_dir='data/assets',asset_id='demo',
        dataset_repo_id='demo',lerobot_root='data/lerobot'))
    assert cfg.model.pi05
    assert cfg.model.action_horizon==50
    assert cfg.model.action_dim==32
    assert cfg.model.max_token_len==200
    assert cfg.model.motion_num_tokens==0
    assert cfg.model.paligemma_variant=='gemma_2b_lora'
    assert cfg.data.adapt_to_pi is False


def test_runtime_preserves_existing_credentials_and_data_cache(tmp_path,monkeypatch):
    from worldla.runtime import configure_runtime
    from openpi.shared.runtime_dirs import configure_runtime_dirs
    monkeypatch.setenv('HF_HOME',str(tmp_path/'user-hf'))
    monkeypatch.setenv('OPENPI_DATA_HOME',str(tmp_path/'user-openpi'))
    configure_runtime(tmp_path/'runtime')
    configure_runtime_dirs()
    assert Path(__import__('os').environ['HF_HOME'])==tmp_path/'user-hf'
    assert Path(__import__('os').environ['OPENPI_DATA_HOME'])==tmp_path/'user-openpi'


def test_source_has_no_development_directory_imports():
    root=Path(__file__).resolve().parents[1]/'src'
    for path in root.rglob('*.py'):
        text=path.read_text()
        assert 'experiments.' not in text,str(path)
        assert '/data'+'1/xuhuixin' not in text,str(path)
        assert '/data'+'2/xuhuixin' not in text,str(path)
