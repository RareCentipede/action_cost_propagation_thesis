import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, List, Tuple, cast
from scipy.spatial import KDTree

from eas.EAS import Domain, State
from eas.block_domain import Pose, Object, Robot
from mapping.oc_map import OccupancyGridMap
from mapping.path_planner import astar, create_nx_nodes

def spawn_blocks(domain: Domain) -> Tuple[Dict[str, Object], List[Tuple[float, float, float]]]:
    """
        Spawn 'real' blocks at positions specified in the state.
        Spawn 'virtual' blocks at their goal positions.
    """
    virtual_blocks = []
    blocks = domain.things.get(Object, []).copy()
    poses = domain.things.get(Pose, [])

    blocks = cast(List[Object], blocks)
    poses = cast(List[Pose], poses)

    for block in blocks:
        goal_pose = block.goal
        if not goal_pose:
            continue

        virtual_block = Object(name=f"{block.name}_virtual", at=goal_pose, real=False)
        virtual_blocks.append(virtual_block)

    blocks.extend(virtual_blocks)
    block_positions = [cast(Pose, block.at).pos for block in blocks]
    blocks_obj_dict = {block.name: block for block in blocks}

    return blocks_obj_dict, block_positions

def perform_initial_cost_propagation(blocks_obj_dict: Dict[str, Object], block_positions: List[Tuple[float, float, float]]) -> np.ndarray:
    real_blocks = [block for block in blocks_obj_dict.values() if block.real]
    scaled_projected_vecs_lists = []

    for block_idx, (block, block_pos) in enumerate(zip(real_blocks, block_positions)):
        init_pos = block_pos
        if not block.goal:
            continue
        goal_pos = block.goal.pos
        goal_pos_id = block_positions.index(goal_pos)

        init_goal_vec = np.array(goal_pos) - np.array(init_pos)

        other_block_positions = block_positions.copy()
        other_block_positions.remove(block_pos)
        other_block_positions.remove(goal_pos)

        dists, projected_vecs_scaling_factors, scaled_projected_vecs = compute_dists_from_point_to_vec(np.array(other_block_positions), init_goal_vec, np.array(init_pos))

        influence_radius = 0.5
        for i in range(len(dists)):
            dist = dists[i]
            scaling = projected_vecs_scaling_factors[i]
            blocking_block_id = i if i < block_idx else i+1
            blocking_block_id = blocking_block_id if blocking_block_id < goal_pos_id else blocking_block_id+1

            print(f"Block {block.name} checking block {blocking_block_id+1}: dist={dist}, scaling={scaling}")

            if dist < influence_radius and (scaling >= 0 and scaling <= 1):
                propagated_cost = (influence_radius*2 - dist)
                blocking_block = list(blocks_obj_dict.values())[blocking_block_id]

                if blocking_block.real:
                    block.propagated_cost += propagated_cost
                else:
                    father_block_name = blocking_block.name.replace("_virtual", "")
                    father_block = blocks_obj_dict[father_block_name]
                    father_block.propagated_cost += propagated_cost

        scaled_projected_vecs_lists.append(scaled_projected_vecs)
    
    return np.array(scaled_projected_vecs_lists)

def compute_dists_from_point_to_vec(points: np.ndarray, vector: np.ndarray, start_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Compute the shortest distances from a set of points to a vector.
        1. Project each point onto the vector
        2. Scale the original vector based on the projection.
        3. Compute the difference between the vector to the point and the scaled vector on the original vector.
        4. Compute the norm of the resultant vectors as the distance.
    """
    init_to_other_blocks_vecs = [np.array(other_pos) - np.array(start_point) for other_pos in points]
    vecs_projected_on_init_goal_vec = np.array([np.dot(vec, vector) for vec in init_to_other_blocks_vecs])
    projected_vecs_scaling_factors = vecs_projected_on_init_goal_vec / (np.linalg.norm(vector)**2)

    scaled_projected_vecs = [scaling * vector for scaling in projected_vecs_scaling_factors]
    dists = [np.linalg.norm(vec - proj_vec) for vec, proj_vec in zip(init_to_other_blocks_vecs, scaled_projected_vecs)]

    return np.array(dists), np.array(projected_vecs_scaling_factors), np.array(scaled_projected_vecs)

def visualize_cost_propagation(blocks_obj_dict: Dict[str, Object], block_positions: List[Tuple[float, float, float]], scaled_projected_vecs_lists: np.ndarray,
                               robot_pos: Tuple[float, float, float] | None = None):
    real_blocks = [block for block in blocks_obj_dict.values() if block.real]

    idx = 0
    for block, block_pos in zip(real_blocks, block_positions):
        if not block.goal:
            continue

        goal_pos = block.goal.pos

        fig = plt.figure()
        ax = fig.add_subplot()

        print(f"Plotting for {block.name}")

        for block_to_plot, block_pos_to_plot in zip(real_blocks, block_positions):
            if robot_pos:
                ax.scatter(robot_pos[0], robot_pos[1], s=50, c='red', marker='^')

            ax.scatter(block_pos_to_plot[0], block_pos_to_plot[1], s=50, c='blue')
            ax.text(block_pos_to_plot[0], block_pos_to_plot[1], f"{block_to_plot.name}\nCost: {block_to_plot.propagated_cost:.1f}", fontsize=8, ha='right')

            if not block_to_plot.goal:
                continue
            goal_pos_to_plot = block_to_plot.goal.pos

            ax.scatter(goal_pos_to_plot[0], goal_pos_to_plot[1], s=50, c='green')
            ax.plot([block_pos_to_plot[0], goal_pos_to_plot[0]], [block_pos_to_plot[1], goal_pos_to_plot[1]], c='black', linestyle='--')

        other_block_positions = block_positions.copy()
        other_block_positions.remove(block_pos)
        other_block_positions.remove(goal_pos)

        scaled_projected_vecs = scaled_projected_vecs_lists[idx]
        for other_block_pos, scaling in zip(other_block_positions, scaled_projected_vecs):
            projected_point = np.array(block_pos) + scaling
            projection = projected_point - np.array(other_block_pos)
            ax.arrow(other_block_pos[0], other_block_pos[1], projection[0], projection[1],
                    head_width=0.05, head_length=0.1, fc='orange', ec='orange', linestyle=':')
            ax.scatter(projected_point[0], projected_point[1], s=30, c='red', marker='x')

        idx += 1

        plt.show()
