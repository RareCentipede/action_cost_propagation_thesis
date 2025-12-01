from collections import deque
import time
import numpy as np

from eas.eas_parser import parse_configs
from eas.block_domain import Pose, Robot, Object, domain, create_goal_nodes, create_domain_transition_graph
from eas.EAS import apply_action, parse_action_params, is_action_applicable, query_current_nodes, query_nodes
from eas.EAS import State, Node, Domain, LinkedState, state_state
from typing import Tuple, Dict, cast, List

def main():
    config_name = "stacked"
    problem_config_path = "config/problem_configs/"

    block_domain = parse_configs(domain, config_name, problem_config_path)
    dtg = create_domain_transition_graph(block_domain)
    goal_nodes = create_goal_nodes(block_domain, dtg)
    print(f"Goal nodes: {[node.name for node in goal_nodes.values()]}")
    current_state = block_domain.current_state

    goal_blocks = [g_node.values[1].name for g_node in goal_nodes.values()]
    goal_positions = [g_node.values[-1].name for g_node in goal_nodes.values()]

    state_counter = 0
    s0 = LinkedState(state=current_state, state_id=state_counter)
    goal_linked_states = []

    current_linked_state = s0
    current_nodes = query_nodes(dtg, current_linked_state.state)

    robot = block_domain.things.get(Robot, [])[0]
    robot = cast(Robot, robot)
    robot_pos = robot.at.name

    block_pos = [cast(Object, obj).at for obj in block_domain.things.get(Object, [])]
    block_pos = [cast(Pose, pos).name for pos in block_pos if pos is not None]
    current_nodes = prune_unrelated_nodes(current_nodes, goal_blocks, block_pos, robot_pos)
    possible_actions = unpack_actions_from_nodes(current_nodes, goal_positions, block_pos, robot_pos)
    current_linked_state.branches_to_explore = possible_actions
    branching = True

    print([[b[0].name, b[1], b[2].name] for b in current_linked_state.branches_to_explore])
    while current_linked_state.branches_to_explore:
        # Print the current tree structure
        # print_tree(s0, current_linked_state)
        block_pos = [cast(Object, obj).at for obj in block_domain.things.get(Object, [])]
        block_pos = [cast(Pose, pos).name for pos in block_pos if pos is not None]

        print(f"Current state id: {current_linked_state.state_id}")
        current_state = current_linked_state.state
        branch = current_linked_state.branches_to_explore.pop(0)

        print(f"Branching: {branch[0].name, branch[1], branch[2].name}")

        node, action_name, target_node = branch
        action_params = parse_action_params(action_name, node, target_node)

        action_tuple = block_domain.actions.get(action_name)
        action_tuple = cast(Tuple, action_tuple)
        _, conds, effects = action_tuple

        action_applicable = is_action_applicable(conds, action_params)
        if action_applicable:
            print(f"Applying action {action_name} from {node.name} to {target_node.name}")
            s_new = apply_action(current_state, conds, action_params, effects)

            ancestor = current_linked_state.parent
            if ancestor:
                if s_new == ancestor.state:
                    print("New state is the same as an ancestor state, skipping to avoid cycle.")
                    branching = False
                elif (action_name, current_linked_state) in ancestor.edges:
                    print("This action from this state was already used to reach the parent state, skipping to avoid cycle.")
                    branching = False
                else:
                    branching = True
            else:
                branching = True

            if branching:
                state_counter += 1
                print(f"Unique state {state_counter} found")
                s_new_linked = LinkedState(state_counter, s_new, parent=current_linked_state)
                current_linked_state.edges.append((action_name, s_new_linked))

                block_domain.update_state(s_new)
                current_linked_state = s_new_linked
                if block_domain.goal_reached:
                    current_linked_state.type_ = state_state.GOAL
                    goal_linked_states.append(s_new_linked)
                    print(f"Goal reached at state id {s_new_linked.state_id}!")
                    break
                else:
                    current_nodes = query_nodes(dtg, current_linked_state.state)
                    robot_pos = robot.at.name
                    current_nodes = prune_unrelated_nodes(current_nodes, goal_blocks, block_pos, robot_pos)
                    possible_actions = unpack_actions_from_nodes(current_nodes, goal_positions, block_pos, robot_pos)
                    current_linked_state.branches_to_explore = possible_actions
        else:
            print(f"Action [{action_name}] not applicable")

        # Backtrack to somewhere with unexplored branches
        while (not current_linked_state.branches_to_explore) or (current_linked_state.type_ == state_state.GOAL):
            if current_linked_state.parent is None:
                print("Explored all branches from the root state.")
                branching = False
                block_domain.update_state(current_linked_state.state)
                break

            print(f"Back track from {current_linked_state.state_id} to {current_linked_state.parent.state_id}")
            current_linked_state = current_linked_state.parent
            block_domain.update_state(current_linked_state.state)
        print("-----------------------------------")
        # time.sleep(0.5)
    # print([g.state for g in goal_linked_states])

def print_tree(root: LinkedState, current: LinkedState | None = None) -> None:
    """
        Breadth-first print of the current search tree.
        Marks the current node and node types (ALIVE/GOAL/DEAD).
    """
    q = deque([root])
    seen: set[int] = set()
    print("\nCurrent tree:")
    while q:
        node = q.popleft()
        if node.state_id in seen:
            continue
        seen.add(node.state_id)

        node_mark = " <== current" if current is not None and node is current else ""
        node_status = "" if node.type_ == state_state.ALIVE else f" [{node.type_.name}]"
        print(f"S{node.state_id}{node_status}{node_mark}")

        for action, child in node.edges:
            child_mark = " <== current" if current is not None and child is current else ""
            child_status = "" if child.type_ == state_state.ALIVE else f" [{child.type_.name}]"
            print(f"  └─[{action}]→ S{child.state_id}{child_status}{child_mark}")
            q.append(child)

def unpack_actions_from_nodes(nodes: List[Node], goal_pos: List[str], block_pos: List[str], robot_pos: str) -> List[Tuple[Node, str, Node]]:
    possible_actions = []

    for node in nodes:
        for edge in node.edges:
            action_name, target_node = edge

            split_base_node_name = node.name.split('_')
            split_target_node_name = target_node.name.split('_')
            base = split_base_node_name[0]
            base_pos = split_base_node_name[-1]

            target_pos = split_target_node_name[-1]

            if base == 'robot' and (target_pos not in block_pos and target_pos not in goal_pos):
                continue
            elif base != 'robot' and ((target_pos == 'None' and base_pos != robot_pos) or (target_pos != 'None' and target_pos not in goal_pos)):
                continue

            possible_actions.append((node, action_name, target_node))

    return possible_actions

def prune_unrelated_nodes(nodes: List[Node], goal_blocks: List[str], block_pos: List[str], robot_pos: str) -> List[Node]:
    for node in nodes:
        split_node_name = node.name.split('_')
        block = split_node_name[0]
        pos = split_node_name[-1]

        if block != 'robot' and ((pos != robot_pos) and (pos != 'None') or block not in goal_blocks):
            nodes.remove(node)

    return nodes

if __name__ == "__main__":
    main()