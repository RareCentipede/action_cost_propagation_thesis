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

    while alive_states:
        alive_state = alive_states.pop(0)

        if alive_state.type_ != state_state.ALIVE:
            continue

        s_current = alive_state.state
        block_domain.update_state(s_current)
        states_in_branch = [s_current]

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

        # print(f"Expanding state {alive_state.state_id} with nodes {[node.name for node in current_nodes]}")
        print(f"Number of edges to explore: {len(alive_state.edges)}")
        alive = False
        while current_nodes:
            node = current_nodes.pop(0)

            for edge in node.edges:
                action_name, to_node = edge
                action_params = parse_action_params(action_name, node, to_node)
                action = block_domain.actions.get(action_name)
                action = cast(Tuple, action)

                _, conds, effects = action

                action_applicable = is_action_applicable(conds, action_params)
                if not action_applicable:
                    continue

                s_new = apply_action(s_current, conds, action_params, effects)
                if s_new in states_in_branch:
                    continue

                # print(f"Applying action {action_name} to reach node {to_node.name} resulting in new state.")

                ancestor = alive_state.parent
                if ancestor is None or ((action_name, alive_state) not in ancestor.edges):
                    state_counter += 1
                    s_new_linked = LinkedState(state_counter, s_new, parent=alive_state)
                    alive_state.edges.append((action_name, s_new_linked))
                    alive_states.append(s_new_linked)

                    states_in_branch.append(s_new)
                    alive = True
                    block_domain.update_state(s_new)

                    # new_nodes = query_nodes(dtg, s_new)
                    # print(f"Reached new state {state_counter} with nodes {[n.name for n in new_nodes]} via action {action_name}")

                    if block_domain.goal_reached:
                        print("Goal reached!")
                        s_new_linked.type_ = state_state.GOAL
                        goal_linked_states.append(s_new_linked)
                        break

                    block_domain.reset_state()

        if not alive:
            alive_state.type_ = state_state.DEAD
            ancestor = alive_state.parent
            if ancestor is not None:
                if len(ancestor.edges) == 1:
                    ancestor.type_ = state_state.DEAD

        if len(alive_state.edges) == 0:
            print("No further actions possible, marking state as DEAD.")
            alive_state.type_ = state_state.DEAD

if __name__ == "__main__":
    main()