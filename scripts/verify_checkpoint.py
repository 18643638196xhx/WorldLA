#!/usr/bin/env python3
"""Compare installed-package inference against saved fixed-input reference outputs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True,
                        help='Batched NPZ: state [B,14], three cameras [B,3,H,W], prompt [B]')
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import jax
    import jax.numpy as jnp
    import openpi
    import worldla
    from openpi.models.model import Observation
    from openpi.shared import nnx_utils
    from worldla.policy import WorldLAPolicy
    print(f'WorldLA package: {worldla.__file__}',flush=True)
    print(f'OpenPI package: {openpi.__file__}',flush=True)
    policy=WorldLAPolicy(args.model_dir)
    normalized=[]
    sample=policy.sample
    def capture(*a,**kw):
        result=sample(*a,**kw)
        normalized.append(np.asarray(result[0]))
        return result
    policy.sample=capture
    route_fn=nnx_utils.module_jit(policy.model.predict_route)
    observations=[]
    with np.load(args.observations,allow_pickle=False) as data:
        for i in range(len(data['state'])):
            observations.append({'state':data['state'][i],'prompt':str(data['prompt'][i]),
                'images':{camera:data[camera][i] for camera in ('cam_high','cam_left_wrist','cam_right_wrist')}})
    actions,routes=[],[]
    for i,observation in enumerate(observations):
        actions.append(policy.infer(observation)['actions'])
        values=policy.inputs(dict(observation))
        obs=Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],values))
        routes.append(np.asarray(route_fn(obs)[0]))
        print(f'Fixed-input sample {i+1}/{len(observations)} complete',flush=True)
    policy.reset()
    plans=np.stack([policy.plan(observation) for observation in observations[:2]])
    actual={'normalized':np.stack(normalized[:len(observations)]),'actions':np.stack(actions),
            'routes':np.stack(routes),'plans':plans}
    comparisons={}
    with np.load(args.reference,allow_pickle=False) as reference:
        for key,value in actual.items():
            delta=np.asarray(value,np.float64)-np.asarray(reference[key],np.float64)
            comparisons[key]={'exact':bool(np.array_equal(value,reference[key])),
                'rmse':float(np.sqrt(np.mean(delta**2))),'max_abs':float(np.max(np.abs(delta)))}
    report={'worldla_package':str(Path(worldla.__file__).resolve()),
            'openpi_package':str(Path(openpi.__file__).resolve()),'observations':len(observations),
            'comparisons':comparisons,'rollout_sr_test':False,
            'all_outputs_finite':all(np.isfinite(x).all() for x in actual.values())}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not policy.config['lora_merged'] and not all(x['exact'] for x in comparisons.values()):
        raise RuntimeError('Original-graph inference differs from saved reference')


if __name__=='__main__':
    main()
