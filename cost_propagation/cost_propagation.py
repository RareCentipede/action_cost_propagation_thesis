import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, cast

from eas.EAS import Domain
from eas.block_domain import Pose, Object

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
        if not goal_pose or (cast(Pose, block.at).pos == goal_pose.pos):
            continue

        virtual_block = Object(name=f"{block.name}_virtual", at=goal_pose, real=False)
        virtual_blocks.append(virtual_block)

    blocks.extend(virtual_blocks)
    block_positions = [cast(Pose, block.at).pos for block in blocks]
    blocks_obj_dict = {block.name: block for block in blocks}

    return blocks_obj_dict, block_positions

def perform_cost_propagation(blocks_obj_dict: Dict[str, Object], block_positions: List[Tuple[float, float, float]]) -> Tuple[List, Dict[str, float]]:
    real_blocks = [block for block in blocks_obj_dict.values() if block.real]
    real_block_positions = [cast(Pose, block.at).pos[:2] for block in real_blocks]
    block_positions_2d = [pos[:2] for pos in block_positions]
    scaled_projected_vecs_lists = []
    propagated_cost_dict = {}
    block_block_name = ""
    p_dict_key = ""

    for block in real_blocks:
        propagated_cost_dict.update({block.name: 0.0})

    for block_idx, (block, block_pos) in enumerate(zip(real_blocks, real_block_positions)):
        init_pos = block_pos
        if not block.goal:
            continue
        goal_pos = block.goal.pos[:2]
        goal_pos_id = block_positions_2d.index(goal_pos)

        init_goal_vec = np.array(goal_pos) - np.array(init_pos)

        other_block_positions = block_positions_2d.copy()
        other_block_positions.remove(block_pos)

        if block_pos != goal_pos:
            other_block_positions.remove(goal_pos)

        dists, projected_vecs_scaling_factors, scaled_projected_vecs = compute_dists_from_point_to_vec(np.array(other_block_positions), init_goal_vec, np.array(init_pos))

        influence_radius = 1.0
        for i in range(len(dists)):
            dist = dists[i]
            scaling = projected_vecs_scaling_factors[i]
            blocking_block_id = i if i < block_idx else i+1
            blocking_block_id = blocking_block_id if blocking_block_id < goal_pos_id else blocking_block_id+1

            if dist < influence_radius and (scaling > 0 and scaling < 1):
                propagated_cost = (influence_radius - dist).item()
                blocking_block = list(blocks_obj_dict.values())[blocking_block_id]

                if blocking_block.real:
                    propagated_cost_dict[block.name] += propagated_cost
                    block_block_name = blocking_block.name
                    p_dict_key = block.name
                else:
                    father_block_name = blocking_block.name.replace("_virtual", "")
                    propagated_cost_dict[father_block_name] += propagated_cost
                    block_block_name = block.name
                    p_dict_key = father_block_name

                # print(f"Block {block_block_name} propagates cost {propagated_cost:.1f} to {p_dict_key}{'_virtual' if not blocking_block.real else ''} \
                #     dist: {dist}, scaling: {scaling}, (total cost: {propagated_cost_dict[p_dict_key]:.1f})")

        scaled_projected_vecs_lists.append(scaled_projected_vecs)
    
    return scaled_projected_vecs_lists, propagated_cost_dict

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

    vector_mag = np.linalg.norm(vector)
    if vector_mag <= 1e-6:
        projected_vecs_scaling_factors = np.zeros(len(init_to_other_blocks_vecs))
    else:
        projected_vecs_scaling_factors = vecs_projected_on_init_goal_vec / (vector_mag**2)

    scaled_projected_vecs = [scaling * vector for scaling in projected_vecs_scaling_factors]
    dists = [np.linalg.norm(vec - proj_vec) for vec, proj_vec in zip(init_to_other_blocks_vecs, scaled_projected_vecs)]

    return np.array(dists), np.array(projected_vecs_scaling_factors), np.array(scaled_projected_vecs)

def visualize_cost_propagation(blocks_obj_dict: Dict[str, Object], block_positions: List[Tuple[float, float, float]], propagated_cost_dict: Dict[str, float], 
                               scaled_projected_vecs_lists: List, robot_pos: Tuple[float, float, float] | None = None):
    real_blocks = [block for block in blocks_obj_dict.values() if block.real]

    idx = 0
    for block, block_pos in zip(real_blocks, block_positions):
        if not block.goal:
            continue
        goal_pos = block.goal.pos

        fig = plt.figure()
        ax = fig.add_subplot()

        for block_to_plot, block_pos_to_plot in zip(real_blocks, block_positions):
            if robot_pos:
                ax.scatter(robot_pos[0], robot_pos[1], s=50, c='red', marker='^')

            ax.scatter(block_pos_to_plot[0], block_pos_to_plot[1], s=50, c='blue')
            p_cost = propagated_cost_dict.get(block_to_plot.name, 0.0)
            ax.text(block_pos_to_plot[0], block_pos_to_plot[1], f"{block_to_plot.name}\nCost: {p_cost:.1f}", fontsize=8, ha='right')

            if not block_to_plot.goal:
                continue
            goal_pos_to_plot = block_to_plot.goal.pos

            ax.scatter(goal_pos_to_plot[0], goal_pos_to_plot[1], s=50, c='green')
            ax.plot([block_pos_to_plot[0], goal_pos_to_plot[0]], [block_pos_to_plot[1], goal_pos_to_plot[1]], c='black', linestyle='--')

        other_block_positions = block_positions.copy()
        other_block_positions.remove(block_pos)

        if block_pos != goal_pos:
            other_block_positions.remove(goal_pos)

        scaled_projected_vecs = scaled_projected_vecs_lists[idx]
        for other_block_pos, scaling in zip(other_block_positions, scaled_projected_vecs):
            projected_point = np.array(block_pos)[:2] + scaling
            projection = projected_point - np.array(other_block_pos)[:2]
            ax.arrow(other_block_pos[0], other_block_pos[1], projection[0], projection[1],
                    head_width=0.05, head_length=0.1, fc='orange', ec='orange', linestyle=':')
            ax.scatter(projected_point[0], projected_point[1], s=30, c='red', marker='x')

        idx += 1

        plt.show()
