from eas.EAS import State
from eas.block_domain import Object, Pose, create_goal_nodes, domain, create_domain_transition_graph
from eas.eas_parser import parse_configs, build_physical_relations
from planners.basic_planner import solve_dtg_basic

def main():
    config_name = "stack_2_stack"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)
    goal_nodes = create_goal_nodes(block_domain, dtg)

    print(type(list(block_domain.current_state.values())[0]))
    # plan = solve_dtg_basic(goal_nodes, dtg, block_domain)
    # for step in plan:
        # print(step)

if __name__ == "__main__":
    main()