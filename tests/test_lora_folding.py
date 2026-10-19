import numpy as np
import pytest
from worldla.checkpoints import fold_lora, validate_tree

@pytest.mark.parametrize('kind', ['q_einsum', 'kv_einsum', 'attn_vec_einsum'])
def test_attention_folding_matches_exact_source_equations(kind):
    rng = np.random.default_rng(42)
    leading = (2, 1) if kind == 'kv_einsum' else (3,)
    w = rng.normal(size=(*leading, 4, 5)).astype(np.float32)
    a = rng.normal(size=(*leading, 4, 2)).astype(np.float32)
    b = rng.normal(size=(*leading, 2, 5)).astype(np.float32)
    parent = ('base', 'PaliGemma', 'llm', 'layers', 'attn', kind)
    output, _ = fold_lora({parent+('w',): w}, {parent+('lora_a',): a, parent+('lora_b',): b}, {'vision_language': .75, 'action': 1.0})
    if kind == 'attn_vec_einsum':
        x = rng.normal(size=(2, 7, 3, 4)).astype(np.float32)
        original = np.einsum('btnh,nhd->btd', x, w) + .75 * np.einsum('btl,nld->btd', np.einsum('btnh,nhl->btl', x, a), b)
        merged = np.einsum('btnh,nhd->btd', x, output[parent+('w',)])
        naive = np.einsum('btnh,nhd->btd', x, w+.75*np.matmul(a,b))
        assert not np.allclose(original, naive, atol=1e-3)
    elif kind == 'kv_einsum':
        x = rng.normal(size=(2, 7, 4)).astype(np.float32)
        original = np.einsum('bsd,ckdh->cbskh', x, w) + .75*np.einsum('cbskl,cklh->cbskh', np.einsum('bsd,ckdl->cbskl', x,a),b)
        merged = np.einsum('bsd,ckdh->cbskh', x,output[parent+('w',)])
    else:
        x = rng.normal(size=(2, 7, 4)).astype(np.float32)
        original = np.einsum('btd,ndh->btnh', x,w) + .75*np.einsum('btnl,nlh->btnh',np.einsum('btd,ndl->btnl',x,a),b)
        merged = np.einsum('btd,ndh->btnh',x,output[parent+('w',)])
    np.testing.assert_allclose(original, merged, rtol=2e-5, atol=2e-5)


def test_scanned_mlp_and_router_are_preserved():
    rng=np.random.default_rng(2)
    key=('base','PaliGemma','llm','layers','mlp_1','gating_einsum')
    w=rng.normal(size=(3,2,4,5)).astype(np.float32)
    a=rng.normal(size=(3,2,4,2)).astype(np.float32)
    b=rng.normal(size=(3,2,2,5)).astype(np.float32)
    delta={key[:-1]+('gating_einsum_lora_a',):a,key[:-1]+('gating_einsum_lora_b',):b,
           ('stage_relation_router','stage_embedding'):np.ones((4,8)),
           ('stage_message_adapter','down','bias'):None}
    output, reports=fold_lora({key:w},delta,{'vision_language':.5,'action':.25})
    np.testing.assert_allclose(output[key],w+np.matmul(a,b))
    assert reports[0]['scale']==1.0  # source FFN has no alpha/rank multiplier
    assert output[('stage_message_adapter','down','bias')] is None
    np.testing.assert_array_equal(output[('stage_relation_router','stage_embedding')],np.ones((4,8)))
    assert not any('lora' in str(key) for key in output)


def test_missing_pair_or_parameter_shape_fails():
    with pytest.raises(ValueError, match='Missing LoRA'):
        fold_lora({('base','q_einsum','w'):np.zeros((4,4))},
                  {('base','q_einsum','lora_a'):np.zeros((4,2))},{'vision_language':1.,'action':1.})
    with pytest.raises(ValueError,match='Parameter keys differ'):
        validate_tree({('a',):np.zeros(2)}, {})
