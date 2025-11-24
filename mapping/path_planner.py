import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from networkx.algorithms.shortest_paths import astar_path
from typing import List, Tuple, Dict, cast
from mapping.oc_map import OccupancyGridMap
from scipy.spatial import KDTree

def create_nx_nodes(ocm: OccupancyGridMap) -> nx.Graph:
    g = nx.Graph()
    grid = ocm.grid
    oc_grid = ocm.oc_grid

    grid_tree = KDTree(grid)

    for point in grid:
        g.add_node((point[0], point[1]))

    for node, occ in zip(g.nodes, oc_grid):
        if occ == 1:
            continue

        neighbors_idx = grid_tree.query(node, k=8)[1].tolist()
        for n_idx in neighbors_idx: #type: ignore
            neighbor = tuple(grid[n_idx])
            if oc_grid[n_idx] == 1 or neighbor == node:
                continue

            g.add_edge(node, neighbor, cost=np.linalg.norm(np.array(node) - np.array(neighbor)))

    return g

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return np.linalg.norm(np.array(a) - np.array(b)).item()

def astar(g: nx.Graph, oc_grid: np.ndarray, start: Tuple[float, float], goal: Tuple[float, float]) -> np.ndarray:
    path = np.array([])
    graph_tree = KDTree(list(g.nodes))

    if start not in g.nodes:
        g.add_node(start)

    if goal not in g.nodes:
        g.add_node(goal)

    start_neighbor_idx = graph_tree.query(start, k=8)[1]
    goal_neighbor_idx = graph_tree.query(goal, k=8)[1]

    for s_idx in start_neighbor_idx: #type: ignore
        s_neighbor = tuple(list(g.nodes)[s_idx])
        if s_neighbor == start or oc_grid[s_idx] == 1:
            continue

        g.add_edge(start, s_neighbor, cost=np.linalg.norm(np.array(start) - np.array(s_neighbor)))

    for g_idx in goal_neighbor_idx: #type: ignore
        g_neighbor = tuple(list(g.nodes)[g_idx])
        if g_neighbor == goal or oc_grid[g_idx] == 1:
            continue
        g.add_edge(goal, g_neighbor, cost=np.linalg.norm(np.array(goal) - np.array(g_neighbor)))

    path = np.array(astar_path(g, start, goal, heuristic=dist, weight='cost'))

    return path