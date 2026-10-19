import numpy as np
import pytest

import flax.nnx as nnx
import jax.numpy as jnp
from worldla.model import StageRelationRouter,StageMessageAdapter,message_source_stage_ids


def test_router_masking_and_normalization():
    model=StageRelationRouter(prefix_dim=8,hidden_dim=4,num_route_tokens=6,num_stages=4,
                              position_only=False,rngs=nnx.Rngs(0))
    mask=jnp.array([[True,True,True,True,True,False,True,True]])
    route=np.asarray(model(jnp.ones((1,8,8)),mask))
    assert route.shape==(1,4,6)
    np.testing.assert_allclose(route.sum(-1),1,rtol=1e-5)
    assert np.all(route[:,:,-1]==0)


def test_zero_initialized_adapter_is_identity():
    model=StageMessageAdapter(prefix_dim=8,suffix_dim=4,bottleneck_dim=2,num_stages=4,rngs=nnx.Rngs(1))
    suffix=jnp.arange(200,dtype=jnp.float32).reshape(1,50,4)
    result=model(suffix,jnp.ones((1,6,8)),jnp.ones((1,4,6))/6)
    np.testing.assert_array_equal(result,suffix)
    ids=np.asarray(message_source_stage_ids(50,4,'normal'))
    np.testing.assert_array_equal(ids,np.arange(50)*4//50)


@pytest.mark.parametrize('stage,condition,expected',[
    ('router','wam_stage_dynamic',{'stage_relation_router'}),
    ('action','wam_stage_dynamic',{'base','stage_message_adapter'}),
    ('action','router_action_only',{'base','stage_relation_router','stage_message_adapter'}),
    ('action','base_lora',{'base'}),
])
def test_trainable_filters(stage,condition,expected):
    from flax import traverse_util
    from openpi.shared import nnx_utils
    from worldla.training import training_filter
    class Model(nnx.Module):
        def __init__(self):
            self.base=nnx.Dict(lora_a=nnx.Param(jnp.ones(1)),weight=nnx.Param(jnp.ones(1)))
            self.stage_relation_router=nnx.Linear(2,2,rngs=nnx.Rngs(0))
            self.stage_message_adapter=nnx.Linear(2,2,rngs=nnx.Rngs(0))
    selected=nnx.state(Model(),training_filter(nnx,nnx_utils,stage=stage,condition=condition))
    leaves=traverse_util.flatten_dict(selected.to_pure_dict())
    assert {key[0] for key in leaves}==expected
    assert ('base','weight') not in leaves
