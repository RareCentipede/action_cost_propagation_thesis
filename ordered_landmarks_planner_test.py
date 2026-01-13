from typing import cast
from planners.ordered_landmarks_planner import OrderedLandmarksPlanner, verbose_levels, heuristic_types
from eas.eas_parser import parse_configs
from eas.block_domain import domain, create_domain_transition_graph
from dispatcher.dispatcher import CommandDispatcher

def main():
    config_name = "stacked"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)

    ap = OrderedLandmarksPlanner(block_domain, dtg, verbose_levels.NONE)
    ap.run_ordered_landmarks_planner(heuristic_types.GREEDY_NEIGHBOR)

    plan = ap.retrace_action_sequence_back_to_root()[0]

    if plan:
        print(f"Plan found 😄! Total number of goal states: {len(ap.goal_linked_states)}")
        cd = CommandDispatcher(block_domain)
        cd.initialize_objects()
        cd.run_simulation(plan)
    else:
        print("No plan found 😢")

if __name__ == "__main__":
    main()