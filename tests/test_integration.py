from types import ModuleType,SimpleNamespace
import json
import sys
import numpy as np

from worldla.integrations.robotwin import observation_from_robotwin,eval,reset_model


def test_robotwin_camera_mapping_and_chunk_execution():
    observation={'observation':{key:{'rgb':np.full((4,5,3),i,np.uint8)} for i,key in
                 enumerate(['head_camera','left_camera','right_camera'])},
                 'joint_action':{'vector':np.zeros(14)}}
    converted=observation_from_robotwin(observation,'pick')
    assert converted['images']['cam_high'].shape==(3,4,5)
    assert converted['images']['cam_left_wrist'].mean()==1
    assert converted['images']['cam_right_wrist'].mean()==2
    class Env:
        eval_success=False
        take_action_cnt=0
        step_lim=3
        def get_instruction(self): return 'pick'
        def take_action(self,action): self.take_action_cnt+=1
    resets=[]
    model=SimpleNamespace(plan=lambda obs:np.ones((10,14)),initial_seed=42,episode_index=-1,
                          reset=lambda *,seed:resets.append(seed))
    env=Env()
    eval(env,model,observation)
    assert env.take_action_cnt==3
    reset_model(model)
    reset_model(model)
    assert resets==[42,1051]


def test_eval_command_preserves_compatibility_root(tmp_path,monkeypatch,capsys):
    from worldla.integrations.evaluate import main
    external=tmp_path/'external_eval.py'
    external.write_text('')
    root=tmp_path/'runtime'
    (root/'script').mkdir(parents=True)
    script=root/'script/eval_policy.py'
    script.symlink_to(external)
    config=tmp_path/'deploy.yaml'
    config.write_text('policy_name: worldla_policy\n')
    monkeypatch.setattr(sys,'argv',['eval-robotwin','--robotwin-root',str(root),
        '--deploy-config',str(config),'--model-dir',str(tmp_path/'model'),
        '--task','press_stapler','--run-tag','test_smoke','--dry-run'])
    main()
    payload=json.loads(capsys.readouterr().out)
    command=payload['command']
    assert str(script) in command
    assert str(external) not in command
    for key,value in [('ckpt_setting','test_smoke'),('train_config_name','worldla_pi05'),
                      ('model_name','worldla')]:
        assert command[command.index('--'+key)+1]==value


def test_renderer_binds_scene_and_renderer_to_visible_cuda(monkeypatch):
    from worldla.integrations.rendering import bind_visible_renderer
    calls=[]
    class BaseRenderer:
        def __init__(self,device): calls.append(('renderer',device))
    class LegacyRenderer(BaseRenderer):
        pass
    device=SimpleNamespace(pci_string='0000:da:00.0')
    class Engine:
        pass
    legacy=ModuleType('sapien.core')
    legacy.SapienRenderer=LegacyRenderer
    legacy.Engine=Engine
    sapien=ModuleType('sapien')
    sapien.core=legacy
    sapien.Device=lambda name:(calls.append(('device',name)) or device)
    sapien.pysapien=SimpleNamespace(render=SimpleNamespace(SapienRenderer=BaseRenderer))
    sapien.physx=SimpleNamespace(set_scene_config=lambda config:calls.append(('config',config)),
                                PhysxCpuSystem=lambda:'physics')
    sapien.render=SimpleNamespace(RenderSystem=lambda selected:('render',selected))
    sapien.Scene=lambda systems:systems
    monkeypatch.setitem(sys.modules,'sapien',sapien)
    monkeypatch.setitem(sys.modules,'sapien.core',legacy)
    assert bind_visible_renderer()=='0000:da:00.0'
    legacy.SapienRenderer()
    assert Engine().create_scene('test_config')==['physics',('render',device)]
    assert calls==[('device','cuda:0'),('renderer',device),('config','test_config')]
