from copy import deepcopy
import random
import numpy as np

from typing import Any, Sequence, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar, cast

from eas.EAS import (Thing, State, SimpleCondition, Domain, Node,
                     is_action_applicable, apply_action, parse_action_params,
                     ComputedCondition, Condition)
from eas.block_domain import Robot, Pose, Object, NonePose, NoneObj, block_domain, create_domain_transition_graph

p1 = Pose(name="p1", pos=(0, 0, 0))
p2 = Pose(name="p2", pos=(0, 0, 0))
p3 = Pose(name="p3", pos=(1, 1, 0))
p4 = Pose(name="p4", pos=(1, 1, 0))
p5 = Pose(name="p5", pos=(2, 2, 0))
p6 = Pose(name="p6", pos=(2, 2, 0))
p7 = Pose(name="p7", pos=(3, 3, 0))
p8 = Pose(name="p8", pos=(3, 3, 0))

robot = Robot(name="robot1", at=p1)
block_1 = Object(name="block1", at=p2)
block_2 = Object(name="block2", at=p3)
block_3 = Object(name="block3", at=p4)

things_init = {
    Robot: [robot],
    Pose: [p1, p2, p3, p4, p5],
    Object: [block_1, block_2, block_3],
}

init_state = State({})
for thing_list in things_init.values():
    for thing in thing_list:
        init_state.update(thing.state)

goal_state = State({
    'block1_at': p6,
    'block2_at': p7,
    'block3_at': p8
})

block_domain.things = things_init
block_domain.states.append(init_state)
block_domain.goal_state = goal_state
block_domain.map_name_to_things()

mock_goal_state = deepcopy(init_state)
mock_goal_state.update({
    'block1_at': p6,
    'block2_at': p7,
    'block3_at': p8
})
# block_domain.update_state(mock_goal_state)

move_conditions = block_domain.actions['move'][1]
move_effects = block_domain.actions['move'][2]

move_conditions = List[SimpleCondition], move_conditions
move_effects = List[SimpleCondition], move_effects

move_params = {
    'robot': robot,
    'start_pose': robot.at,
    'target_pose': p5
}

dtg = create_domain_transition_graph(block_domain)
goal_nodes = {}
for var, val in block_domain.goal_state.items():
    dtg_key = f"{var}_{val}"
    goal_node = dtg.get(dtg_key, None)

    if goal_node:
        goal_nodes[dtg_key] = goal_node

while not block_domain.goal_reached:
    current_nodes = []
    for var, val in block_domain.current_state.items():
        dtg_key = f"{var}_{val}"
        current_node = dtg.get(dtg_key, None)
        # print(f"dtg_key: {var}_{val}, current_node: {current_node.name if current_node else None}")
        if current_node:
            current_nodes.append(current_node)

    node_action_values = {}
    print([node.name for node in current_nodes])

    for node_id, node in enumerate(current_nodes):
        action_values = []
        print([[e[0], e[1].name] for e in node.edges])

        for edge in node.edges:
            # print(edge[0])
            action_name, target = edge
            action_value = 0

            action = block_domain.actions.get(action_name)

            if not action:
                continue

            _, conds, effects = action

            action_params = parse_action_params(action_name, node, target)

            # if action_name == 'pick':
            #     print(action_params, conds)

            action_applicable = is_action_applicable(conds, action_params)

            if not action_applicable:
                action_values.append(-1)
                continue

            if target in goal_nodes.values():
                action_value += 1

            actions_in_target = [e[0] for e in target.edges]
            if 'pick' in actions_in_target:
                action_value += 2
            elif 'place' in actions_in_target:
                action_value += 3

            action_values.append(action_value)

        node_action_values[node_id] = [np.argmax(action_values), max(action_values)]

    # print(f"node_action_values: {node_action_values}")
    best_node = np.argmax(np.array(list(node_action_values.values()))[:, 1])
    action_id = node_action_values[best_node][0]
    edge = current_nodes[best_node].edges[action_id]

    action_name, target = edge
    action_params = parse_action_params(action_name, current_nodes[best_node], target)

    action = block_domain.actions.get(action_name)
    action = cast(Tuple, action)
    _, conds, effects = action

    print(f"{current_nodes[best_node].name} --[{action_name}]--> {target.name} Executed")
    new_state = apply_action(block_domain.current_state, conds, action_params, effects)
    block_domain.update_state(new_state)
    # print(f"current state: {new_state}")
        # for edge in node.edges:
        #     action_name, target = edge

        #     action = block_domain.actions.get(action_name)

        #     if not action:
        #         continue

        #     _, conds, effects = action

        #     # print(action_name, node.name, target.name)
        #     action_params = parse_action_params(action_name, node, target)
        #     action_applicable = is_action_applicable(conds, action_params)

        #     if action_applicable:
        #         print(f"{node.name} --[{action_name}]--> {target.name} Executed")
        #         new_state = apply_action(block_domain.current_state, conds, action_params, effects)
        #         block_domain.update_state(new_state)
                # print(block_domain.current_state)