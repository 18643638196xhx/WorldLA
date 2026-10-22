import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from worldla.integrations.xpolicylab import Model, actions_to_xpolicylab, observation_from_xpolicylab
from worldla.policy import WorldLAPolicy


def observation(env_idx=0, value=0):
    return {
        'env_idx': env_idx, 'instruction': 'Place the can in the basket.',
        'vision': {name: {'color': np.full((4, 5, 3), i, np.uint8)} for i, name in
                   enumerate(('cam_head', 'cam_left_wrist', 'cam_right_wrist'))},
        'state': {name: np.full(size, value, np.float32) for name, size in
                  (('left_arm_joint_state', 6), ('left_ee_joint_state', 1),
                   ('right_arm_joint_state', 6), ('right_ee_joint_state', 1))},
    }


class FakePolicy:
    plan = WorldLAPolicy.plan

    def __init__(self, directory, seed):
        self.root = Path(directory)
        self.initial_seed = seed
        self.model = object()
        self.sample = object()
        self.config = {'lora_merged': False, 'deployment': {'execute_k': 10, 'old_plan_weight': .75}}
        self.reset(seed=seed)

    def reset(self, *, seed):
        self.rng = seed
        self.previous_plan = None

    def infer(self, obs):
        self.rng += 1
        commands = np.arange(50, dtype=np.float32)[:, None] + obs['state'][0]
        return {'actions': np.repeat(commands, 14, axis=1)}


@pytest.fixture
def model(monkeypatch, tmp_path):
    monkeypatch.setattr('worldla.policy.WorldLAPolicy', FakePolicy)
    return Model({'model_dir': tmp_path, 'inference_seed': 42})


def repack(actions):
    return np.stack([np.concatenate([row[k] for k in
        ('left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state', 'right_ee_joint_state')])
        for row in actions])


def test_standard_observation_preserves_rgb_joint_order_and_readonly_arrays():
    obs = observation()
    for i, name in enumerate(obs['state']):
        obs['state'][name][...] = i
    rgb = obs['vision']['cam_head']['color']
    rgb[0, 0] = (255, 17, 3)
    rgb.flags.writeable = False
    result = observation_from_xpolicylab(obs)
    np.testing.assert_array_equal(result['state'], [0]*6 + [1] + [2]*6 + [3])
    np.testing.assert_array_equal(result['images']['cam_high'][:, 0, 0], (255, 17, 3))
    assert result['images']['cam_left_wrist'].shape == (3, 4, 5)
    assert result['images']['cam_right_wrist'].mean() == 2
    assert result['state'].dtype == np.float32


@pytest.mark.parametrize('bad', ['image_float', 'image_encoded', 'state_shape', 'state_nan', 'prompt'])
def test_invalid_observations_fail_instead_of_silent_conversion(bad):
    obs = observation()
    if bad == 'image_float': obs['vision']['cam_head']['color'] = np.zeros((4, 5, 3), np.float32)
    if bad == 'image_encoded': obs['vision']['cam_head']['color'] = b'encoded jpeg bytes'
    if bad == 'state_shape': obs['state']['left_arm_joint_state'] = np.zeros(7)
    if bad == 'state_nan': obs['state']['left_arm_joint_state'][0] = np.nan
    if bad == 'prompt': obs['instruction'] = ''
    with pytest.raises(ValueError): observation_from_xpolicylab(obs)


def test_actions_are_full_joint_dictionaries_without_scaling_or_clipping():
    commands = np.arange(140, dtype=np.float32).reshape(10, 14)
    actions = actions_to_xpolicylab(commands, 10)
    np.testing.assert_array_equal(repack(actions), commands)
    assert all(set(row) == set(observation()['state']) for row in actions)
    actions[0]['left_arm_joint_state'][0] = -999
    assert commands[0, 0] == 0
    with pytest.raises(ValueError): actions_to_xpolicylab(commands[:9], 10)


def test_single_environment_uses_original_plan_fusion_and_episode_seed(model):
    with pytest.raises(ValueError): model.get_action()
    model.reset()
    model.update_obs(observation())
    first = repack(model.get_action())
    np.testing.assert_array_equal(first[:, 0], np.arange(10))
    assert model._sessions[0].rng == 43
    with pytest.raises(ValueError): model.get_action()
    model.update_obs(observation(value=100))
    second = repack(model.get_action())
    expected = .75*np.arange(10, 20) + .25*np.arange(100, 110)
    np.testing.assert_array_equal(second[:, 0], expected)
    np.testing.assert_array_equal(second[:, 6], np.arange(10, 20))  # discrete old gripper
    model.reset()
    assert not model._sessions and not model._observations
    model.update_obs(observation())
    model.get_action()
    assert model._sessions[0].rng == 42 + 1009 + 1


def test_batches_reorder_shrink_and_keep_plan_rng_separate(model):
    model.reset()
    model.update_obs_batch([observation(9, 100), observation(3, 200)])
    chunks = model.get_action_batch([3, 9])
    assert repack(chunks[0])[0, 0] == 200 and repack(chunks[1])[0, 0] == 100
    a, b = model._sessions[3], model._sessions[9]
    assert a is not b and a.model is b.model is model.policy.model
    assert a.sample is b.sample and a.previous_plan is not b.previous_plan
    assert a.rng != b.rng
    b_previous = b.previous_plan.copy()
    model.update_obs_batch([observation(3, 300)])
    out = model.get_action_batch()
    assert len(out) == 1 and repack(out[0])[0, 0] == .75*210 + .25*300
    np.testing.assert_array_equal(b.previous_plan, b_previous)
    model.update_obs_batch([observation(9, 400)])
    assert repack(model.get_action())[0, 0] == .75*110 + .25*400
    model.on_trial_end({'success': True})
    assert not model._sessions


def test_invalid_batch_is_atomic_and_requires_explicit_ids(model):
    model.update_obs_batch([observation(5), observation(2)])
    with pytest.raises(ValueError): model.get_action()
    with pytest.raises(ValueError): model.get_action_batch([5, 99])
    assert not model._sessions
    with pytest.raises(ValueError): model.get_action_batch([5, 5])
    with pytest.raises(ValueError): model.update_obs_batch([observation(1), observation(1)])
    with pytest.raises(ValueError): model.update_obs_batch([observation(-1)])
    assert model._order == [5, 2]


@pytest.mark.parametrize('config', [
    {'action_type': 'ee'}, {'old_plan_weight': 2}, {'inference_seed': -1},
    {'robot_action_dim_info': {'arm_dim': [7, 7], 'ee_dim': [1, 1]}},
])
def test_unsupported_configuration_rejected_before_weight_load(config, tmp_path):
    with pytest.raises(ValueError): Model({'model_dir': tmp_path, **config})


def test_server_dry_run_is_lazy(tmp_path, monkeypatch, capsys):
    from worldla.integrations.xpolicylab_server import main
    root = tmp_path/'XPolicyLab'
    (root/'client_server/ws').mkdir(parents=True)
    (root/'client_server/ws/model_server.py').write_text('')
    model = tmp_path/'full'
    model.mkdir()
    (model/'config.json').write_text(json.dumps({'schema': 'worldla_full_original_v1',
        'lora_merged': False, 'base_model_config': {'paligemma_variant': 'gemma_2b_lora',
                                                 'action_expert_variant': 'gemma_300m_lora'}}))
    monkeypatch.setattr(sys, 'argv', ['serve', '--xpolicylab-root', str(root),
        '--model-dir', str(model), '--dry-run'])
    main()
    data = json.loads(capsys.readouterr().out)
    assert not data['model_loaded'] and not data['server_started']


def test_server_initializes_policy_before_optional_native_transport(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from worldla.integrations import xpolicylab_server
    root = tmp_path/'XPolicyLab'
    (root/'client_server/ws').mkdir(parents=True)
    (root/'client_server/ws/model_server.py').write_text('')
    model = tmp_path/'full'
    model.mkdir()
    (model/'config.json').write_text(json.dumps({'schema': 'worldla_full_original_v1',
        'lora_merged': False, 'base_model_config': {'paligemma_variant': 'gemma_2b_lora',
                                                 'action_expert_variant': 'gemma_300m_lora'}}))
    order = []
    monkeypatch.setattr('worldla.integrations.xpolicylab.Model', lambda cfg: order.append('model'))
    class Server:
        def __init__(self, model, config): pass
        def serve_forever(self): return 'server token'
    def api(path):
        order.append('transport')
        return Server, SimpleNamespace
    monkeypatch.setattr(xpolicylab_server, 'load_server_api', api)
    monkeypatch.setattr(xpolicylab_server.asyncio, 'run', lambda token: order.append(token))
    monkeypatch.setattr(sys, 'argv', ['serve', '--xpolicylab-root', str(root), '--model-dir', str(model)])
    xpolicylab_server.main()
    assert order == ['model', 'transport', 'server token']


@pytest.mark.parametrize('task_config,instruction', [('demo_clean', 'seen'), ('demo_randomized', 'unseen')])
def test_native_evaluator_dry_run_uses_separate_python(tmp_path, monkeypatch, capsys, task_config, instruction):
    from worldla.integrations.xpolicylab_eval import main
    root = tmp_path/'RoboTwin'
    (root/'scripts').mkdir(parents=True)
    (root/'scripts/eval_policy_xpolicylab.py').write_text('')
    (root/'env_cfg/task_config').mkdir(parents=True)
    (root/'env_cfg/task_config'/f'{task_config}.yml').write_text('')
    monkeypatch.setattr(sys, 'argv', ['eval', '--robotwin-root', str(root), '--task', 'press_stapler',
        '--env-python', sys.executable, '--task-config', task_config, '--episodes', '3', '--dry-run'])
    main()
    data = json.loads(capsys.readouterr().out)
    cmd = data['command']
    assert cmd[0] == sys.executable
    assert cmd[cmd.index('--test_num')+1] == '3'
    assert cmd[cmd.index('--instruction_type')+1] == instruction
    assert '--eval_episode_num' not in cmd and '--overrides' not in cmd
    assert cmd[cmd.index('--eval_batch')+1] == 'false'
    assert not data['starts_policy_server']


def test_dropin_adapter_imports_package_and_runs_complete_chunks():
    root = Path(__file__).resolve().parents[1]/'integrations/xpolicylab/WorldLA'
    spec = importlib.util.spec_from_file_location('dropin_worldla_model', root/'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.Model is Model
    spec = importlib.util.spec_from_file_location('dropin_worldla_deploy', root/'deploy.py')
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    class Env:
        def __init__(self): self.steps = 0
        def is_episode_end(self): return self.steps >= 23
        def get_obs(self): return {'step': self.steps}
        def take_action(self, action): self.steps += 1
    calls = []
    class Client:
        def call(self, func_name, obs=None):
            calls.append((func_name, copy.deepcopy(obs)))
            return [{}]*10 if func_name == 'get_action' else None
    env = Env()
    deploy.eval_one_episode(TASK_ENV=env, model_client=Client())
    assert env.steps == 23
    assert [obs['step'] for name, obs in calls if name == 'update_obs'] == [0, 10, 20]
