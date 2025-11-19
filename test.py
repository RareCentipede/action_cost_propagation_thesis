from eas.block_domain import Pose, Robot, Object, Ground

p1 = Pose("p1", (0.0, 0.0, 0.0))
p2 = Pose("p2", (1.0, 0.0, 0.0))
p3 = Pose("p3", (2.0, 0.0, 0.0))
b1 = Object("b1", p1, False, Ground())
b2 = Object("b2", p2, True, b1)
b1.below = b2

setattr(b2, 'on.at_top', True)
print(b1.at_top)