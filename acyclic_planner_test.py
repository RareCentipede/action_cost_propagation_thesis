from planners.acyclic_planner import AcyclicPlanner, verbose_levels
from eas.eas_parser import parse_configs
from eas.block_domain import  domain, create_domain_transition_graph

def main():
    config_name = "stacked"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)

    ap = AcyclicPlanner(block_domain, dtg)
    ap.run_acyclic_planner()

if __name__ == "__main__":
    main()