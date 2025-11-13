import numpy as np

from yaml import safe_load
from typing import Dict, List, Tuple, cast

from eas.EAS import Domain, State
from eas.block_domain import Robot, Pose, Object

def parse_configs(domain: Domain, config_name: str, problem_config_path: str = "config/problem_configs/") -> Domain:
    init_config, goal_config = load_configs_to_dict(config_name, problem_config_path)
    define_init_objects_and_poses(init_config, domain)
    define_goal_objects_and_poses(goal_config, domain)
    initialize_states_and_domain(domain)

    return domain

def load_configs_to_dict(config_name: str, problem_config_path: str) -> Tuple[Dict, Dict]:
    init_path = problem_config_path + config_name + "/init.yaml"
    goal_path = problem_config_path + config_name + "/goal.yaml"

    with open(init_path, 'r') as f:
        init_config = safe_load(f)
        f.close()

    with open(goal_path, 'r') as f:
        goal_config = safe_load(f)
        f.close()

    return init_config, goal_config

def define_init_objects_and_poses(init_config: Dict, domain: Domain):
    idx = 1
    for obj_name, info in init_config.items():
        obj_type = obj_name.split('_')[0]

        pose_name = 'p' + str(idx)
        pose = Pose(pose_name, info['position'])

        if obj_type == 'robot':
            robot = Robot(obj_name, at=pose)
            domain.things.setdefault(Robot, []).append(robot)
        else:
            block = Object(obj_name, at=pose)
            domain.things.setdefault(Object, []).append(block)

        domain.things.setdefault(Pose, []).append(pose)

        idx += 1

def define_goal_objects_and_poses(goal_config: Dict, domain: Domain):
    goal_state = State({})
    for obj_name, info in goal_config.items():
        pos = info['position']

        poses = domain.things.get(Pose, [])
        poses = cast(List[Pose], poses)
        pose = find_pose_from_position(pos, poses)
        goal_state.update({f"{obj_name}_at": pose.name})

        if pose not in domain.things.get(Pose, []):
            domain.things.setdefault(Pose, []).append(pose)

    domain.goal_state = goal_state

def find_pose_from_position(pos: Tuple[float, float, float], poses: List[Pose]) -> Pose:
    positions = [pose.pos for pose in poses]

    for idx, position in enumerate(positions):
        if np.allclose(np.array(position), np.array(pos)):
            return poses[idx]

    pose_name = 'p' + str(len(poses) + 1)
    pose = Pose(pose_name, pos)
    poses.append(pose)
    return pose

def initialize_states_and_domain(domain: Domain):
    init_state = State({})
    for thing_list in domain.things.values():
        for thing in thing_list:
            init_state.update(thing.state)

    domain.states.append(init_state)
    domain.map_name_to_things()