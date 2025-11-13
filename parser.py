from yaml import safe_load
from typing import Dict, Tuple, cast

from eas.EAS import Domain, State
from eas.block_domain import Robot, Pose, Object, create_goal_nodes, domain, create_domain_transition_graph
from planners.basic_planner import solve_dtg_basic

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
        poses = [cast(Pose, pose).pos for pose in poses]
        if pos not in poses:
            pose_name = 'p' + str(len(poses) + 1)
            pose = Pose(pose_name, pos)
            domain.things.setdefault(Pose, []).append(pose)

        goal_state.update({f"{obj_name}_at": pose_name})

    domain.goal_state = goal_state

def initialize_states_and_domain(domain: Domain):
    init_state = State({})
    for thing_list in domain.things.values():
        for thing in thing_list:
            init_state.update(thing.state)

    domain.states.append(init_state)
    domain.map_name_to_things()

def main():
    config_name = "stacked"
    problem_config_path = "config/problem_configs/"

    init_config, goal_config = load_configs_to_dict(config_name, problem_config_path)
    define_init_objects_and_poses(init_config, domain)
    define_goal_objects_and_poses(goal_config, domain)
    initialize_states_and_domain(domain)

    dtg = create_domain_transition_graph(domain)
    goal_nodes = create_goal_nodes(domain, dtg)

    plan = solve_dtg_basic(goal_nodes, dtg, domain)
    for step in plan:
        print(step)

if __name__ == "__main__":
    main()