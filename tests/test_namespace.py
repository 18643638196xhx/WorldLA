"""Namespace/API changes must preserve existing checkpoint parameter layouts."""
import ast
import json
from pathlib import Path
import sys


def test_no_legacy_namespace_imports_or_package_directories():
    root=Path(__file__).resolve().parents[1]
    assert not (root/'src/openpi').exists()
    assert not (root/'src/openpi_client').exists()
    for path in (root/'src/worldla').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.ImportFrom): modules=[node.module or '']
            elif isinstance(node,ast.Import): modules=[a.name for a in node.names]
            assert not any(name.split('.')[0] in ('openpi','openpi_client') for name in modules),str(path)
            if isinstance(node,ast.ClassDef):
                assert node.name not in {'Pi0','Pi0Config','Pi0FAST','Pi0FASTConfig',
                                         'PI0Pytorch','StageRelationRouterPi','StageRouteOpenPiDataset'},str(path)


def test_public_api_and_full_checkpoint_parameter_layout():
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    import jax
    from worldla import WorldLAConfig,WorldLAModel
    from worldla.models import WorldLABackbone
    cfg=WorldLAConfig(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',pi05=True)
    shape=nnx.eval_shape(lambda:WorldLAModel(cfg.create(jax.random.key(0)),
        prefix_dim=2048,suffix_dim=1024,num_route_tokens=768,num_stages=4,
        router_hidden_dim=256,adapter_dim=128,router_active=True,message_enabled=True,
        position_only=False,rngs=nnx.Rngs(0)))
    assert isinstance(shape.base,WorldLABackbone)
    layout={'/'.join(map(str,key)):None if value is None else list(value.shape)
            for key,value in flatten_dict(nnx.state(shape).to_pure_dict()).items()}
    expected=json.loads((Path(__file__).parent/'fixtures/original_parameter_layout.json').read_text())['layout']
    assert layout==expected
    assert not any(name=='openpi' or name.startswith('openpi.') or name=='openpi_client'
                   or name.startswith('openpi_client.') for name in sys.modules)
