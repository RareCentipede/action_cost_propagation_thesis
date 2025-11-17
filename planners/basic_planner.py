import time
import numpy as np

from eas.block_domain import Pose, Robot, Object
from eas.EAS import apply_action, parse_action_params, is_action_applicable
from eas.EAS import State, Node, Domain
from typing import Tuple, Dict, cast, List

def query_current_nodes(dtg: Dict[str, Node], current_state: State, goal_nodes: Dict[str, Node]) -> List[Node]:
    current_nodes = []
    for var, val in current_state.items():
        dtg_key = f"{var}_{val}"
        current_node = dtg.get(dtg_key, None)
        if current_node and current_node not in goal_nodes.values():
            current_nodes.append(current_node)
    return current_nodes

def compute_action_values(node: Node, goal_nodes: Dict[str, Node], actions: Dict[str, Tuple],
                          current_block_positions: List[Object], goal_blocks: List[Object], goal_positions: List[Pose]) -> List:
    action_values = []
    for edge in node.edges:
        action_name, target = edge
        action_value = 0

        action = actions.get(action_name)

        if not action:
            continue

        _, conds, _ = action

        action_params = parse_action_params(action_name, node, target)
        action_applicable = is_action_applicable(conds, action_params)
        if not action_applicable:
            action_values.append(-1)
            continue

        if target in goal_nodes.values():
            action_value += 5

        robot = action_params.get('robot')
        robot = cast(Robot, robot)

        if action_name == 'move':
            if target.values[1] in goal_positions and not robot.gripper_empty and target.values[1].occupied_by == "NoneObject":
                action_value += 4
            elif target.values[1] in current_block_positions and robot.gripper_empty and target.values[1] not in goal_positions:
                action_value += 2
            else:
                action_value += 1
        elif action_name == 'pick':
            obj = action_params.get('object')
            obj = cast(Object, obj)
            if obj in goal_blocks and robot.gripper_empty:
                action_value += 3

        action_values.append(action_value)

    return action_values

def apply_best_action(node_action_values: Dict, current_nodes: List[Node], domain: Domain) -> Tuple[State, List[str]]:
    valid_node_actions = {k: v for k, v in node_action_values.items() if v[1] >= 0}
    valid_actions = [av for av in valid_node_actions.values()]
    new_state = State({})
    plan = []

    for node_id, action_values in node_action_values.items():
        print(f"Node: {current_nodes[node_id].name}, Best Action ID: {action_values[0]}, Value: {action_values[1]}")

    while valid_actions:
        best_node = np.argmax(np.array(valid_actions)[:, 1])
        best_node_key = list(valid_node_actions.keys())[best_node]
        action_id = valid_node_actions[best_node_key][0]
        edge = current_nodes[best_node_key].edges[action_id]

        action_name, target = edge
        action_params = parse_action_params(action_name, current_nodes[best_node_key], target)

        action = domain.actions.get(action_name)
        action = cast(Tuple, action)
        _, conds, effects = action

        action = f"{current_nodes[best_node_key].name} --[{action_name}]--> {target.name}"
        plan.append(action)
        print(action)
        new_state = apply_action(domain.current_state, conds, action_params, effects)
        if len(domain.states) >= 2:
            if new_state == domain.states[-2]:
                print("Reverted to previous state, choosing next best action...\n")
                valid_actions.pop(best_node)
                continue
            else:
                break
        else:
            break

    return new_state, plan

def solve_dtg_basic(goal_nodes: Dict[str, Node], dtg: Dict[str, Node], domain: Domain) -> List[str]:
    goal_blocks = [g_node.values[1] for g_node in goal_nodes.values()]
    goal_positions = [g_node.values[-1] for g_node in goal_nodes.values()]
    actions_in_domain = domain.actions
    actions = []

    while not domain.goal_reached:
        current_state = domain.current_state
        current_nodes = query_current_nodes(dtg, current_state, goal_nodes)
        current_block_positions = [node.values[-1] for node in current_nodes if type(node.values[1]) == Object]

        node_action_values = {}
        # print(f"Current nodes: {[node.name for node in current_nodes]}")

        for node_id, node in enumerate(current_nodes):
            action_values = compute_action_values(node, goal_nodes, actions_in_domain,
                                                  current_block_positions, goal_blocks, goal_positions)

            # TODO: Make it able to choose more actions, because sometimes many actions have the same values, \
                # but not all of them result in a good state
            node_action_values[node_id] = action_values

        new_state, new_actions = apply_best_action(node_action_values, current_nodes, domain)
        actions.extend(new_actions)

        time.sleep(0.1)
        if new_state:
            domain.update_state(new_state)

    print("Goal reached!")
    return actions