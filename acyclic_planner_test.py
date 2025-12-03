from planners.acyclic_planner import AcyclicPlanner, verbose_levels
from eas.eas_parser import parse_configs
from eas.block_domain import  domain, create_domain_transition_graph
from dispatcher.dispatcher import CommandDispatcher

def main():
    config_name = "basic"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)

    ap = AcyclicPlanner(block_domain, dtg)
    ap.run_acyclic_planner(verbose_levels.INFO)

    plan = ap.retrace_action_sequence_back_to_root()

    if plan:
        print("Plan found 😄")
        cd = CommandDispatcher(block_domain)
        cd.initialize_objects()
        cd.run_simulation(plan)
    else:
        print("No plan found 😢")

if __name__ == "__main__":
    main()