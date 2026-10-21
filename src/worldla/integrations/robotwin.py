"""Policy plug-in for RoboTwin's get_model/eval/reset_model interface."""
import numpy as np


def get_model(usr_args):
    from worldla.policy import WorldLAPolicy
    import worldla
    import worldla.models.worldla
    print(f'[worldla] Policy plug-in: {__file__}', flush=True)
    print(f'[worldla] Model implementation: {worldla.models.worldla.__file__}', flush=True)
    print(f'[worldla] Package implementation: {worldla.__file__}', flush=True)
    model = WorldLAPolicy(usr_args['model_dir'],seed=int(usr_args.get('inference_seed',20260901)))
    print(f'[worldla] Full checkpoint loaded: {model.root}; lora_merged={model.config["lora_merged"]}', flush=True)
    weight = float(usr_args.get('old_plan_weight',model.config['deployment']['old_plan_weight']))
    if not 0 <= weight <= 1:
        raise ValueError('old_plan_weight must lie in [0,1]')
    model.config['deployment']['old_plan_weight'] = weight
    model.episode_index = -1
    return model


def observation_from_robotwin(observation, instruction):
    images = observation['observation']
    result = {'state': np.asarray(observation['joint_action']['vector']), 'prompt':str(instruction),
              'images':{}}
    for name,source in [('cam_high','head_camera'),('cam_left_wrist','left_camera'),
                        ('cam_right_wrist','right_camera')]:
        image = np.asarray(images[source]['rgb'])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f'Expected HWC RGB uint8 for {source}')
        result['images'][name] = image.transpose(2,0,1)
    return result


def eval(task_env, model, observation):
    observation = observation_from_robotwin(observation,task_env.get_instruction())
    commands = model.plan(observation)
    for command in commands:
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            break
        task_env.take_action(command)


def reset_model(model):
    # Match the reference evaluator's per-episode sampling seed convention.
    model.episode_index += 1
    model.reset(seed=model.initial_seed + 1009 * model.episode_index)
