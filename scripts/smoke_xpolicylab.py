#!/usr/bin/env python3
"""Test the real official XPolicyLab websocket transport with a toy policy.

No GPU, model weights, simulator, or robot commands are used. The server binds
only an ephemeral loopback port and is closed before this process exits.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from worldla.integrations.xpolicylab import Model
from worldla.integrations.xpolicylab_server import load_server_api
from worldla.policy import WorldLAPolicy


class ToyPolicy:
    plan = WorldLAPolicy.plan

    def __init__(self, directory, seed):
        self.root = Path(directory)
        self.model = object()
        self.sample = object()
        self.initial_seed = seed
        self.config = {'deployment': {'execute_k': 10, 'old_plan_weight': .75}}
        self.reset(seed=seed)

    def reset(self, *, seed):
        self.rng, self.previous_plan = seed, None

    def infer(self, obs):
        self.rng += 1
        result = np.arange(50, dtype=np.float32)[:, None] + obs['state'][0]
        return {'actions': np.repeat(result, 14, axis=1)}


def observation(env_idx=0, value=0, *, encode_image):
    image = np.zeros((16, 16, 3), np.uint8)
    image[..., 0] = 220
    return {
        'instruction': 'Transport smoke test; never execute.', 'env_idx': env_idx,
        'vision': {camera: {'color': encode_image(image.copy())} for camera in
                   ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        'state': {key: np.full(size, value, np.float32) for key, size in
                  (('left_arm_joint_state', 6), ('left_ee_joint_state', 1),
                   ('right_arm_joint_state', 6), ('right_ee_joint_state', 1))},
    }


async def run(root):
    server_class, config_class = load_server_api(root)
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.protocol.exceptions import WsError
    from XPolicyLab.utils.process_data import encode_image_bit

    with patch('worldla.policy.WorldLAPolicy', ToyPolicy):
        model = Model({'model_dir': '.', 'inference_seed': 42})
    server = server_class(model, config_class(host='127.0.0.1', port=0))
    await server.start()

    def exercise():
        with WsModelClient(url=server.url, evaluation_id='worldla-wire-smoke',
                           trial_id='test-only', action_case_id='case-0',
                           request_timeout_s=10, max_connect_attempts=1) as client:
            client.call('prepare_case', obs={'task_name': 'test-only', 'seed': 100000})
            client.call('reset')
            obs = observation(encode_image=encode_image_bit)
            client.call('update_obs', obs=obs)
            actions = client.call('get_action')
            assert len(actions) == 10 and actions[0]['left_arm_joint_state'].shape == (6,)
            assert actions[0]['left_arm_joint_state'][0] == 0
            image = model._observations[0]['images']['cam_high']
            assert image.shape == (3, 16, 16) and image[0].mean() > 200 and image[2].mean() < 10
            client.call('update_obs', obs=observation(value=100, encode_image=encode_image_bit))
            fused = client.call('get_action')
            assert fused[0]['left_arm_joint_state'][0] == .75*10 + .25*100
            print('[worldla/smoke] Next request deliberately reuses an observation; '
                  'an error response is expected.', flush=True)
            try:
                client.call('get_action')  # Must not advance state again without fresh input.
            except WsError:
                pass
            else:
                raise AssertionError('Expected a protocol error for reused observations')
            client.call('reset')
            batch = [observation(9, 20, encode_image=encode_image_bit),
                     observation(4, 30, encode_image=encode_image_bit)]
            client.call('update_obs_batch', obs=batch)
            chunks = client.call('get_action_batch', obs=[4, 9])
            assert chunks[0][0]['left_arm_joint_state'][0] == 30
            assert chunks[1][0]['left_arm_joint_state'][0] == 20
            assert model._sessions[4].model is model._sessions[9].model
            old_9 = copy.deepcopy(model._sessions[9].previous_plan)
            client.call('update_obs_batch', obs=[observation(4, 40, encode_image=encode_image_bit)])
            client.call('get_action_batch', obs=[4])
            np.testing.assert_array_equal(model._sessions[9].previous_plan, old_9)
            client.call('trial_end', obs={'success': True})
            assert not model._sessions
        return {'transport': 'official XPolicyLab websocket (real loopback socket)',
                'encoded_rgb_decoding': True, 'single_chunk_and_fusion': True,
                'batch_id_order_and_state_isolation': True, 'reset_and_trial_end': True,
                'error_round_trip': True, 'real_checkpoint_loaded': False,
                'simulator_evaluated': False}

    try:
        return await asyncio.to_thread(exercise)
    finally:
        await server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xpolicylab-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.xpolicylab_root)), indent=2))


if __name__ == '__main__':
    main()
