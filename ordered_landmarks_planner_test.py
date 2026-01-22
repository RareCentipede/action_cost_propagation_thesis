import numpy as np

from typing import cast
import matplotlib.pyplot as plt

from planners.ordered_landmarks_planner import OrderedLandmarksPlanner, verbose_levels, heuristic_types
from eas.eas_parser import parse_configs
from eas.block_domain import Object, Pose, Robot, domain, create_domain_transition_graph
from dispatcher.dispatcher import CommandDispatcher
from cost_propagation.cost_propagation import spawn_blocks, perform_initial_cost_propagation, visualize_cost_propagation
from mapping.oc_map import OccupancyGridMap
from mapping.path_planner import create_nx_nodes, astar

def main():
    config_name = "basic_many"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)

    oc_map = OccupancyGridMap(block_domain, grid_res=0.5, col_margin=0.5)
    oc_grid = oc_map.create_occupancy_grid_map()
    graph = create_nx_nodes(oc_map)

    blocks_obj_dict, block_positions = spawn_blocks(block_domain)
    scaled_projected_vecs_lists = perform_initial_cost_propagation(blocks_obj_dict, block_positions)

    robot = block_domain.things.get(Robot, [])[0]
    robot = cast(Robot, robot)
    robot_init_pos = cast(Pose, robot.at).pos
    print(f"Robot initial position: {robot_init_pos}")

    fig, ax = plt.subplots(figsize=(8, 8))
    oc_map.plot_occupancy_grid_map(oc_map.grid, oc_grid, ax=ax)
    ax.scatter(robot_init_pos[0], robot_init_pos[1], color='blue', label='Robot Start', s=50, marker='^')

    # plt.show()

    # visualize_cost_propagation(blocks_obj_dict, block_positions, scaled_projected_vecs_lists, robot_pos=robot.at.pos)

    ap = OrderedLandmarksPlanner(block_domain, dtg, oc_map, verbose_levels.DEBUG)
    # ap.run_ordered_landmarks_planner(heuristic_types.LAZY_GREEDY)
    ap.run_optimal_ordered_landmarks_planner(heuristic_types.GREEDY_NEIGHBOR_PROPAGATED)
    # plans, states = ap.retrace_action_sequence_back_to_root()
    # plan = plans[0] if plans else None

    plan, states = ap.retrace_optimal_action_sequence_back_to_root()
    total_path_cost = 0.0

    print(f"num states: {len(states)}")

    if plan:
        print(f"Plan found 😄! Total number of goal states: {len(ap.goal_linked_states)}")

        for step_idx, action in enumerate(plan):
            print(f"Step {step_idx}: {action}")

            current_state = states[step_idx]
            robot_at = current_state[f"{robot.name}_at"]
            robot_pos = block_domain.name_things.get(robot_at)
            robot_pos = cast(Pose, robot_pos).pos

            action_name, params = action
            if action_name == 'move':
                # fig, ax = plt.subplots(figsize=(8, 8))
                oc_grid = oc_map.assign_occupancy_from_state(states[step_idx])
                # oc_map.plot_occupancy_grid_map(oc_map.grid, oc_grid, ax=ax)
                graph = create_nx_nodes(oc_map)

                ax.scatter(robot_pos[0], robot_pos[1], color='blue', label='Robot Start', s=50, marker='^')

                _, start_pose, goal_pose = params
                start_pose = block_domain.name_things.get(start_pose)
                goal_pose = block_domain.name_things.get(goal_pose)
                start_pose = cast(Pose, start_pose)
                goal_pose = cast(Pose, goal_pose)

                start_pos = (start_pose.pos[0], start_pose.pos[1])
                goal_pos = (goal_pose.pos[0], goal_pose.pos[1])
                path = np.array(astar(graph, oc_map.oc_grid, start_pos, goal_pos))
                path_cost = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                total_path_cost += path_cost
                print(f"Path cost: {path_cost:.2f}, Total path cost so far: {total_path_cost:.2f}")

                # ax.plot(path[:,0], path[:,1], color='red')

                # plt.show()

        # fig, ax = plt.subplots(figsize=(8, 8))
        oc_grid = oc_map.assign_occupancy_from_state(states[step_idx])
        # oc_map.plot_occupancy_grid_map(oc_map.grid, oc_grid, ax=ax)
        # ax.scatter(robot_pos[0], robot_pos[1], color='blue', label='Robot Start', s=50, marker='^')

        # plt.show()

        print(f"Total path cost of the plan: {total_path_cost:.2f}")

        cd = CommandDispatcher(block_domain)
        cd.initialize_objects()
        cd.run_simulation(plan)

    else:
        print("No plan found 😢")

if __name__ == "__main__":
    main()