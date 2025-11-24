import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict, cast

def create_nx_nodes(oc_grid: np.ndarray, grid_limits: Tuple[Tuple[float, float], Tuple[float, float]], grid_res: float) -> nx.Graph:
    g = nx.Graph()
    (min_x, max_x), (min_y, max_y) = grid_limits
    x_size, y_size = oc_grid.shape

    for i in range(x_size):
        for j in range(y_size):
            node_id = (i, j)
            if oc_grid[node_id] == 1 or node_id in g.nodes:
                continue

            node_pos = (min_x + i, min_y - j)
            g.add_node(node_pos)

            neighbors = np.array([
                (i-1, j), (i+1, j),
                (i, j-1), (i, j+1),
                (i-1, j-1), (i-1, j+1),
                (i+1, j-1), (i+1, j+1)
            ])
            for neighbor in neighbors:
                if neighbor == node_id or neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] >= x_size or neighbor[1] >= y_size or oc_grid[neighbor] == 1:
                    continue

                if neighbor in g.nodes:
                    g.add_edge(node_id, neighbor)
                else:
                    neighbor_pos = (min_x + neighbor[0], min_y - neighbor[1])
                    g.add_node(neighbor_pos)
                    g.add_edge(node_pos, neighbor_pos)

    return g