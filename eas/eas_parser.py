import numpy as np

from yaml import safe_load
from typing import Dict, List, Tuple, cast
from scipy.spatial import KDTree

from eas.EAS import Domain, State
from eas.block_domain import Robot, Pose, Object, Ground    

def parse_configs(domain: Domain, config_name: str, problem_config_path: str = "config/problem_configs/") -> Domain:
    gnd = Ground()
    domain.things.setdefault(Ground, []).append(gnd)
    domain.name_things[gnd.name] = gnd

    init_config, goal_config = load_configs_to_dict(config_name, problem_config_path)
    define_init_objects_and_poses(init_config, domain)
    define_goal_objects_and_poses(goal_config, domain)
    build_physical_relations(domain)
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
            domain.name_things[obj_name] = robot
        else:
            block = Object(obj_name, at=pose)
            pose.clear = False
            pose.occupied_by = block
            domain.things.setdefault(Object, []).append(block)
            domain.name_things[obj_name] = block

        domain.things.setdefault(Pose, []).append(pose)
        domain.name_things[pose.name] = pose

        idx += 1

def define_goal_objects_and_poses(goal_config: Dict, domain: Domain):
    goal_state = State({})
    for obj_name, info in goal_config.items():
        pos = info['position']

        poses = domain.things.get(Pose, [])
        poses = cast(List[Pose], poses)
        pose = find_pose_from_position(pos, poses)
        goal_state.update({f"{obj_name}_at": pose.name})

        if pose not in poses:
            domain.things.setdefault(Pose, []).append(pose)
            domain.name_things[pose.name] = pose
            print(f"Added new goal pose: {pose}")

    domain.goal_state = goal_state

def find_pose_from_position(pos: Tuple[float, float, float], poses: List[Pose]) -> Pose:
    positions = [pose.pos for pose in poses]

    for idx, position in enumerate(positions):
        if np.allclose(np.array(position), np.array(pos)):
            return poses[idx]

    pose_name = 'p' + str(len(poses) + 1)
    pose = Pose(pose_name, pos)
    return pose

def initialize_states_and_domain(domain: Domain):
    init_state = State({})
    for thing_list in domain.things.values():
        for thing in thing_list:
            print(thing.state)
            init_state.update(thing.state)

    domain.states.append(init_state)

def build_physical_relations(domain: Domain) -> List[List[str]]:
    visited_positions = []
    stacks = []

    poses = domain.things.get(Pose, [])
    poses = cast(List[Pose], poses)
    positions = [pose.pos[:2] for pose in poses]
    pos_tree = KDTree(positions)

    for i, pos in enumerate(positions):
        if i in visited_positions:
            continue

        pos_idx_in_stack = pos_tree.query_ball_point(pos, r=0.05, p=2)
        visited_positions.extend(pos_idx_in_stack)

        poses_in_stack = np.array(poses)[pos_idx_in_stack]
        poses_in_stack = sorted(poses_in_stack, key=lambda p: (p.pos[2]))

        for j, pose in enumerate(poses_in_stack):
            if j == 0:
                pose.on = domain.name_things['GND']

            if j < len(poses_in_stack) - 1:
                above_pose = poses_in_stack[j+1]
                pose.below = above_pose
                above_pose.on = pose

                occupied_obj = pose.occupied_by
                above_obj = above_pose.occupied_by

                if type(occupied_obj) is Object:
                    occupied_obj.at_top = False

                    if j == 0:
                        occupied_obj.on = cast(Ground, domain.name_things['GND'])

                    if type(above_obj) is Object:
                        occupied_obj.below = above_obj
                        above_obj.on = occupied_obj

            elif j == len(poses_in_stack) - 1:
                occupied_obj = pose.occupied_by
                if isinstance(occupied_obj, Object):
                    occupied_obj.at_top = True

        stacks.append([pose.name for pose in poses_in_stack])

    return stacks