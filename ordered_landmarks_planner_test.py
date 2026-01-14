import numpy as np

from typing import cast
from planners.ordered_landmarks_planner import OrderedLandmarksPlanner, verbose_levels, heuristic_types
from eas.eas_parser import parse_configs
from eas.block_domain import Object, Robot, domain, create_domain_transition_graph
from dispatcher.dispatcher import CommandDispatcher
from cost_propagation.cost_propagation import spawn_blocks, perform_initial_cost_propagation, visualize_cost_propagation
from mapping.oc_map import OccupancyGridMap

def main():
    config_name = "basic"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)

    blocks_obj_dict, block_positions = spawn_blocks(block_domain)
    perform_initial_cost_propagation(blocks_obj_dict, block_positions)
    objects = block_domain.things.get(Object, [])

    robot = block_domain.things.get(Robot, [])[0]
    robot = cast(Robot, robot)

    visualize_cost_propagation(blocks_obj_dict, block_positions, robot.at.pos)

    for obj in cast(list[Object], objects):
        print(f"Object: {obj.name}, Propagated Cost: {obj.propagated_cost}")

    ap = OrderedLandmarksPlanner(block_domain, dtg, verbose_levels.NONE)
    ap.run_ordered_landmarks_planner(heuristic_types.GREEDY_NEIGHBOR)

    plan = ap.retrace_action_sequence_back_to_root()[0]

    if plan:
        print(f"Plan found 😄! Total number of goal states: {len(ap.goal_linked_states)}")
        # cd = CommandDispatcher(block_domain)
        # cd.initialize_objects()
        # cd.run_simulation(plan)
    else:
        print("No plan found 😢")

if __name__ == "__main__":
    main()