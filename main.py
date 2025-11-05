from typing import Any, Sequence, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar, cast

from eas.EAS import (Thing, State, SimpleCondition, Domain, Node,
                     is_action_applicable, apply_action, parse_action_params,
                     ComputedCondition, Condition)
from eas.block_domain import Robot, Pose, Object, NonePose, NoneObj, block_domain, create_domatin_transition_graph

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

# mock_goal_state = init_state
# mock_goal_state.update({
#     'block1_at': p6,
#     'block2_at': p7,
#     'block3_at': p8
# })
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

dtg = create_domatin_transition_graph(block_domain)

current_nodes = []
for var, val in block_domain.current_state.items():
    dtg_key = f"{var}_{val}"
    current_node = dtg.get(dtg_key, None)

    if current_node:
        current_nodes.append(current_node)

for node in current_nodes:
    for edge in node.edges:
        action_name, target = edge

        action = block_domain.actions.get(action_name)

        if not action:
            continue

        _, conds, effects = action
        action_params = parse_action_params(action_name, node, target)

        action_applicable = is_action_applicable(conds, action_params)
        print(f"{node.name} --[{action_name}]--> {target.name} | Applicable: {action_applicable}")