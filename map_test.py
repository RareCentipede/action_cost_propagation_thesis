import time
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, cast

from mapping.path_planner import astar
from eas.EAS import State
from eas.block_domain import Object, Pose, domain
from eas.eas_parser import parse_configs
from mapping.oc_map import OccupancyGridMap
from mapping.path_planner import create_nx_nodes
from dispatcher.dispatcher import CommandDispatcher

def main():
    config_name = "basic"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    ocm = OccupancyGridMap(block_domain, grid_res=0.5, col_margin=0.0)
    grid = ocm.create_occupancy_grid_map()
    graph = create_nx_nodes(ocm)

    start = (-6.0, -2.0)
    goal = (0.0, 7.0)

    path = np.array(astar(graph, ocm.oc_grid, start, goal))
    ocm.plot_occupancy_grid_map(ocm.grid, ocm.oc_grid)
    plt.plot(path[:,0], path[:,1], color='red')
    plt.show()
    # cd = CommandDispatcher(block_domain)
    # cd.initialize_objects()
    # time.sleep(10.0)

if __name__ == "__main__":
    main()