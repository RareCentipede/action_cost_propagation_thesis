from typing import cast
from planners.acyclic_planner import AcyclicPlanner, verbose_levels
from eas.eas_parser import parse_configs
from eas.block_domain import  Object, domain, create_domain_transition_graph
from dispatcher.dispatcher import CommandDispatcher

def main():
    config_name = "stacked"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)

    ap = AcyclicPlanner(block_domain, dtg, verbose_levels.INFO)
    ap.run_acyclic_planner()

    plan = ap.retrace_action_sequence_back_to_root()

    if plan:
        print(f"Plan found 😄! Total number of goal states: {len(ap.goal_linked_states)}")
        # cd = CommandDispatcher(block_domain)
        # cd.initialize_objects()
        # cd.run_simulation(plan)
    else:
        print("No plan found 😢")

if __name__ == "__main__":
    main()