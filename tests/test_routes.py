import json
import sys

import numpy as np
import pytest

from worldla.data.build_routes import four_stage_route
from worldla.data.build_routes import main as build
from worldla.data.stage_routes import StageRouteTargetStore
from worldla.data.stage_routes import randomize_stage_targets
from worldla.data.spatial import to_pi_tokens


def test_three_view_mass_is_preserved():
    heat=np.zeros((12,10),np.float32)
    heat[:8]=1
    heat[8:,:5]=2
    heat[8:,5:]=4
    result=to_pi_tokens(heat,16).reshape(3,256)
    expected=np.array([80,40,80])/200
    np.testing.assert_allclose(result.sum(-1),expected,rtol=1e-5)


def test_action_stage_targets_change_without_physical_phase_claim():
    teacher=np.zeros((2,8,120),np.float32)
    for stage in range(4):
        teacher[:,stage*2:(stage+1)*2,stage*20]=1
    route,strength,t,a=four_stage_route({'shared_k_route_abs_ta':teacher},16)
    assert route.shape==(4,768) and (t,a)==(2,8)
    np.testing.assert_allclose(route.sum(-1),1,rtol=1e-5)
    assert np.unique(np.argmax(route,axis=-1)).size==4
    with pytest.raises(ValueError,match='finite'):
        four_stage_route({'shared_k_route_abs_ta':np.full((1,4,120),np.nan)},16)


def test_builder_and_store_work_with_relative_cache_paths(tmp_path,monkeypatch):
    source=tmp_path/'teacher'
    source.mkdir()
    np.savez(source/'cache.npz',shared_k_route_abs_ta=np.ones((2,8,120),np.float32))
    row={'sample_id':'ep0_f0','global_index':0,'episode_index':0,'frame_index':0,
         'split':'train','family':'pick','teacher_npz':'cache.npz'}
    manifest=source/'teacher_manifest.jsonl'
    manifest.write_text(json.dumps(row)+'\n')
    output=tmp_path/'routes'
    monkeypatch.setattr(sys,'argv',['build','--teacher-manifest',str(manifest),'--output-dir',str(output)])
    build()
    row=json.loads((output/'stage_route_manifest.jsonl').read_text())
    assert row['stage_route_npz']=='samples/ep0_f0.npz'
    store=StageRouteTargetStore(output/'stage_route_manifest.jsonl')
    np.testing.assert_allclose(store.target(0,'wam_stage_dynamic').sum(-1),1,rtol=1e-5)


def test_manifest_rejects_episode_leakage(tmp_path):
    path=tmp_path/'manifest.jsonl'
    rows=[{'global_index':i,'episode_index':0,'split':split} for i,split in enumerate(['train','val'])]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError,match='multiple splits'):
        StageRouteTargetStore(path,load_routes=False)


def test_shuffled_teacher_is_deterministic_and_no_stage_is_fixed():
    target=np.tile(np.eye(4,dtype=np.float32)[None],(8,1,1))
    a=randomize_stage_targets(target,np.arange(8),seed=12,step=2)
    np.testing.assert_array_equal(a,randomize_stage_targets(target,np.arange(8),seed=12,step=2))
    assert np.all(np.argmax(a,axis=-1)!=np.arange(4))
