import time
import numpy as np

from eas.eas_parser import parse_configs
from eas.block_domain import Pose, Robot, Object, domain, create_goal_nodes, create_domain_transition_graph
from eas.EAS import apply_action, parse_action_params, is_action_applicable, query_current_nodes, query_nodes
from eas.EAS import State, Node, Domain, LinkedState, state_state
from typing import Tuple, Dict, cast, List

def main():
    config_name = "simple"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)
    goal_nodes = create_goal_nodes(block_domain, dtg)

    current_state = block_domain.current_state

    goal_blocks = [g_node.values[1].name for g_node in goal_nodes.values()]
    goal_positions = [g_node.values[-1].name for g_node in goal_nodes.values()]
    actions_in_domain = domain.actions

    state_counter = 0
    s0 = LinkedState(state=current_state, state_id=state_counter)
    alive_states = [s0]
    goal_linked_states = []

    """
        Things to try:
            - Instead of always creating new linked states, check if the new state already exists in the alive states or goal states.
            - Do backtracking when a state is marked as DEAD or GOAL, and from there explore new branches.
            - Instead of applying all possible actions, choose one and keep track of it, so that when backtracking different actions
                can be tried, and the states are well connected and not duplicated.
    """

    """
        Tree branching idea:
            - Initialization:
                - Create initial linked state s0 with current state.
                - Set current linked state to s0.
            - Loop:
                - From current linked state, query current nodes.
                - Prune nodes that are not relevant to the goal or would lead to cycles.
                - Find all possible actions from current nodes, and add them to branches_to_explore of current linked state.
                - While branches_to_explore is not empty:
                    - Pop an action from branches_to_explore.
                    - Check if action is applicable.
                        - Apply action to get new state s_new.
                        - Check if s_new was reached using the same action as parent from ancestor, if so skip to avoid cycles.
                        - Create new linked state s_new_linked with s_new, set parent to current linked state.
                        - Add (action, s_new_linked) to edges of current linked state.
                        - Update domain state to s_new.
                        - If goal reached:
                            - Mark s_new_linked as GOAL.
                            - Add s_new_linked to goal linked states.
                            - Set mode to backtracking.
                        - Query current nodes from s_new, prune irrelevant nodes, and add possible actions to branches_to_explore of s_new_linked.
                        - Set current linked state to s_new_linked.
 
                    - If branches_to_explore is empty:
                        - Set current linked state to parent of current linked state (backtrack).
                        - If no parent (reached root), terminate loop.
    """

    while alive_states:
        alive_state = alive_states.pop(0)

        s_current = alive_state.state
        block_domain.update_state(s_current)

        robot = block_domain.things.get(Robot, [])[0]
        robot = cast(Robot, robot)
        robot_pos = robot.at.name

        current_nodes = query_nodes(dtg, s_current)

        for node in current_nodes:
            split_node_name = node.name.split('_')
            block = split_node_name[0]
            pos = split_node_name[-1]

            if block == 'robot':
                continue
            else:
                block = block[:-1] # remove the block number at the end

            if pos != robot_pos or block not in goal_blocks:
                current_nodes.remove(node)

        node_idx = 0
        terminate = False
        while not terminate:
            node = current_nodes[node_idx]

            branching = True
            alive = False
            while branching:
                if not node.edges:
                    break

                edge = node.edges.pop(0)
                action_name, target = edge
                action_params = parse_action_params(action_name, node, target)

                action_tuple = block_domain.actions.get(action_name)
                action_tuple = cast(Tuple, action_tuple)

                _, conds, effects = action_tuple        

                action_applicable = is_action_applicable(conds, action_params)
                if not action_applicable:
                    continue
                alive = True

                s_new = apply_action(s_current, conds, action_params, effects)

                ancestor = alive_state.parent
                if ancestor is None or ((action_name, alive_state) not in ancestor.edges):
                    state_counter += 1
                    s_new_linked = LinkedState(state_counter, s_new, parent=alive_state)
                    alive_state.edges.append((action_name, s_new_linked))

                    block_domain.update_state(s_new)

                    if block_domain.goal_reached:
                        print("\nGoal reached!\n")
                        s_new_linked.type_ = state_state.GOAL
                        goal_linked_states.append(s_new_linked)
                        alive = False

                block_domain.reset_state()

if __name__ == "__main__":
    main()