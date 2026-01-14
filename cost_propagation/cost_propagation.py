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

def perform_initial_cost_propagation(blocks_obj_dict: Dict[str, Object], block_positions: List[Tuple[float, float, float]]):
    real_blocks = [block for block in blocks_obj_dict.values() if block.real]

    for block_idx, (block, block_pos) in enumerate(zip(real_blocks, block_positions)):
        init_pos = block_pos
        goal_pose = cast(Pose, block.goal)
        if not goal_pose:
            continue
        goal_pos = goal_pose.pos

        init_goal_vec = np.array(goal_pos) - np.array(init_pos)

        other_block_positions = block_positions.copy()
        other_block_positions.remove(block_pos)

        dists, projected_vecs_scaling_factors = compute_dists_from_point_to_vec(np.array(other_block_positions), init_goal_vec, np.array(init_pos))

        influence_radius = 1.0
        for i in range(len(dists)):
            dist = dists[i]
            scaling = projected_vecs_scaling_factors[i]
            blocking_block_id = i if i < block_idx else i+1

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

def compute_dists_from_point_to_vec(points: np.ndarray, vector: np.ndarray, start_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Compute the shortest distances from a set of points to a vector.
        1. Project each point onto the vector
        2. Scale the original vector based on the projection.
        3. Compute the difference between the vector to the point and the scaled vector on the original vector.
        4. Compute the norm of the resultant vectors as the distance.
    """
    init_to_other_blocks_vecs = [np.array(other_pos) - np.array(start_point) for other_pos in points]
    vecs_projected_on_init_goal_vec = np.array([np.dot(vec, vector) for vec in init_to_other_blocks_vecs])
    projected_vecs_scaling_factors = vecs_projected_on_init_goal_vec / np.linalg.norm(vector)

    scaled_projected_vecs = [scaling * vector for scaling in projected_vecs_scaling_factors]
    dists = [np.linalg.norm(vec - proj_vec) for vec, proj_vec in zip(init_to_other_blocks_vecs, scaled_projected_vecs)]

    return np.array(dists), np.array(projected_vecs_scaling_factors)

def visualize_cost_propagation(blocks_obj_dict: Dict[str, Object], block_positions: List[Tuple[float, float, float]], robot_pos: Tuple[float, float, float]):
    fig = plt.figure()
    ax = fig.add_subplot()

    real_blocks = [block for block in blocks_obj_dict.values() if block.real]

    ax.scatter(robot_pos[0], robot_pos[1], s=50, c='red', marker='^')

    for block, block_pos in zip(real_blocks, block_positions):
        ax.scatter(block_pos[0], block_pos[1], s=50, c='blue')
        ax.text(block_pos[0], block_pos[1], f"{block.name}\nCost: {block.propagated_cost:.1f}", fontsize=8, ha='right')

        goal_pose = block.goal
        if not goal_pose:
            continue
        goal_pos = goal_pose.pos

        ax.scatter(goal_pos[0], goal_pos[1], s=50, c='green')
        ax.plot([block_pos[0], goal_pos[0]], [block_pos[1], goal_pos[1]], c='black', linestyle='--')

    plt.show()
