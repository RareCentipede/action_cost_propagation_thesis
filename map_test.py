from eas.EAS import State
from eas.block_domain import Object, Pose, domain
from eas.eas_parser import parse_configs
from mapping.oc_map import OccupancyGridMap
from dispatcher.dispatcher import CommandDispatcher

def main():
    config_name = "basic"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    ocm = OccupancyGridMap(block_domain, grid_res=0.5)
    grid = ocm.create_occupancy_grid_map()
    ocm.plot_occupancy_grid_map(grid)
    # cd = CommandDispatcher(block_domain)
    # cd.initialize_objects()

if __name__ == "__main__":
    main()