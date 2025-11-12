from eas.EAS import State
from eas.block_domain import Robot, Pose, Object, domain, create_domain_transition_graph
from planners.basic_planner import solve_dtg_basic

p1 = Pose(name="p1", pos=(0, 0, 0))
p2 = Pose(name="p2", pos=(0, 0, 0))
p3 = Pose(name="p3", pos=(1, 1, 0))
p4 = Pose(name="p4", pos=(1, 1, 0))
p5 = Pose(name="p5", pos=(2, 2, 0))
p6 = Pose(name="p6", pos=(2, 2, 0))
p7 = Pose(name="p7", pos=(3, 3, 0))
p8 = Pose(name="p8", pos=(3, 3, 0))
p9 = Pose(name="p9", pos=(4, 4, 0))
p10 = Pose(name="p10", pos=(4, 4, 0))

robot = Robot(name="robot1", at=p1)
block_1 = Object(name="block1", at=p5)
block_2 = Object(name="block2", at=p2)
block_3 = Object(name="block3", at=p9)
block_4 = Object(name="block4", at=p6, on=block_3)

things_init = {
    Robot: [robot],
    Pose: [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10],
    Object: [block_1, block_2, block_3, block_4],
}

init_state = State({})
for thing_list in things_init.values():
    for thing in thing_list:
        init_state.update(thing.state)

goal_state = State({
    'block1_at': 'p3',
    'block2_at': 'p7',
    'block3_at': 'p8',
})

domain.things = things_init
domain.states.append(init_state)
domain.goal_state = goal_state
domain.map_name_to_things()

dtg = create_domain_transition_graph(domain)

goal_nodes = {}
for var, val in domain.goal_state.items():
    dtg_key = f"{var}_{val}"
    goal_node = dtg.get(dtg_key, None)

    if goal_node:
        goal_nodes[dtg_key] = goal_node

plan = solve_dtg_basic(goal_nodes, dtg, domain)
for step in plan:
    print(step)