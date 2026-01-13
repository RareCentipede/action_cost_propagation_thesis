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

    for block, block_pos in zip(real_blocks, block_positions):
        init_pos = block_pos
        goal_pose = cast(Pose, block.goal)
        if not goal_pose:
            continue
        goal_pos = goal_pose.pos

        init_goal_vec = np.array(goal_pos) - np.array(init_pos)
        init_goal_vec_u = init_goal_vec / np.linalg.norm(init_goal_vec)

        other_block_positions = block_positions.copy()
        other_block_positions.remove(block_pos)

        init_to_other_blocks_vecs = [np.array(other_pos) - np.array(init_pos) for other_pos in other_block_positions]
        vector_scaling_factors = np.array([np.dot(init_goal_vec_u, vec) for vec in init_to_other_blocks_vecs])/np.linalg.norm(init_goal_vec)
        projected_vecs = [scaling * init_goal_vec for scaling in vector_scaling_factors]
        dists = [np.linalg.norm(vec - proj_vec) for vec, proj_vec in zip(init_to_other_blocks_vecs, projected_vecs)]

        influence_radius = 0.5
        for i in range(len(dists)):
            dist = dists[i]
            scaling = vector_scaling_factors[i]

            if dist < influence_radius and scaling > 0:
                propagated_cost = (influence_radius*2 - dist).item()
                blocking_block_id = i+1
                blocking_block = list(blocks_obj_dict.values())[blocking_block_id]

                if blocking_block.real:
                    block.propagated_cost += propagated_cost
                else:
                    father_block_name = blocking_block.name.replace("_virtual", "")
                    father_block = blocks_obj_dict[father_block_name]
                    father_block.propagated_cost += propagated_cost