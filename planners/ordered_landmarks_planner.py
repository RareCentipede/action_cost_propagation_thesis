import numpy as np

from collections import deque
from enum import Enum

from eas.block_domain import Pose, Robot, Object, create_goal_nodes
from eas.EAS import Action, Effect, apply_action, parse_action_params, is_action_applicable, query_available_nodes, query_nodes
from eas.EAS import State, Node, Domain, LinkedState, StateStatus, Condition
from typing import Tuple, Dict, cast, List

verbose_levels = Enum('VerboseLevel', 'NONE DEBUG TRACK INFO')

class OrderedLandmarksPlanner:
    def __init__(self, domain: Domain, dtg: Dict[str, Node], verbosity: verbose_levels = verbose_levels.NONE):
        self.domain = domain
        self.dtg = dtg
        self.verbosity = verbosity

        self.goal_nodes = create_goal_nodes(self.domain, self.dtg)
        self.current_state = self.domain.current_state
        self.goal_blocks = [g_node.values[1].name for g_node in self.goal_nodes.values()]
        self.goal_positions = [g_node.values[-1].name for g_node in self.goal_nodes.values()]

        self.state_counter = 0
        self.steps = 0
        self.s0 = LinkedState(state=self.current_state, state_id=self.state_counter)
        self.current_linked_state = self.s0
        self.goal_linked_states = []

        robot = domain.things.get(Robot, [])[0]
        self.robot = cast(Robot, robot)

    def run_ordered_landmarks_planner(self) -> List[LinkedState]:
        goal_linked_states = []
        shortest_num_steps = np.inf

        self.current_linked_state = self.branch_out(self.find_block_positions(), self.current_linked_state)

        # Define branches at current state
        # Evalute the branches if there is more than one
        # Assign the costs to the edges
        # Choose the lowest cost edge to expand next

        # Can potentially have multiple queues for multiple heuristics

        return goal_linked_states

    def branch_out(self, block_pos: List[str], current_linked_state: LinkedState) -> LinkedState:
        """
            Find useful branches to explore from the current state. Return the updated linked state with branches to explore.
        """
        current_nodes = query_available_nodes(self.dtg, current_linked_state.state)
        current_nodes = self.prune_unrelated_nodes(current_nodes)
        preferred_action = self.find_preferred_action(block_pos, current_nodes)

        match preferred_action:
            case 'pick':
                home_node, target_node = self.define_pick_branch(self.robot.at.name)
                branches = [(home_node, preferred_action, target_node)]
            case 'place':
                home_node, target_node = self.define_place_branch()
                branches = [(home_node, preferred_action, target_node)]
            case 'move':
                home_node = self.dtg.get(f"robot_at_{self.robot.at.name}")
                home_node = cast(Node, home_node)

                branches = []
                target_nodes = self.define_move_branches(current_nodes)
                for target_node in target_nodes:
                    branches.append((home_node, preferred_action, target_node))
            case _:
                raise ValueError(f"Unknown action: {preferred_action}")

        weighted_branches = self.evaluate_branches(branches, heuristic='greedy')
        current_linked_state.branches_to_explore = weighted_branches

        return current_linked_state

    def define_pick_branch(self, robot_pos: str) -> Tuple[Node, Node]:
        pos = self.domain.name_things.get(robot_pos)
        pos = cast(Pose, pos)

        block_to_pick = pos.occupied_by
        block_to_pick = cast(Object, block_to_pick)

        node_name = f"{block_to_pick.name}_at_{robot_pos}"                
        home_node = self.dtg.get(node_name)
        home_node = cast(Node, home_node)

        target_node_name = f"{block_to_pick.name}_at_None"
        target_node = self.dtg.get(target_node_name)
        target_node = cast(Node, target_node)

        return home_node, target_node

    def define_place_branch(self) -> Tuple[Node, Node]:
        block_to_place = self.robot.holding
        block_to_place = cast(Object, block_to_place)
        goal_pose = block_to_place.goal
        goal_pose = cast(Pose, goal_pose)

        home_node_name = f"{block_to_place.name}_at_None"
        home_node = self.dtg.get(home_node_name)
        home_node = cast(Node, home_node)

        target_node_name = f"{block_to_place.name}_at_{goal_pose.name}"
        target_node = self.dtg.get(target_node_name)
        target_node = cast(Node, target_node)

        return home_node, target_node

    def define_move_branches(self, nodes: List[Node]) -> List[Node]:
        """
            If gripper emptry, many choices for movement. If gripper already holding an object, only move
            to its goal position.
        """
        target_nodes = []

        if self.robot.gripper_empty:
            poses = [node.values[2] for node in nodes if node.name.startswith('block')]
            positions = [pose.pos for pose in poses]

            robot_position = self.robot.at.pos
            dists_to_poses = np.linalg.norm(np.array(positions) - np.array(robot_position), axis=1)
            closest_pose_idx = np.argmin(dists_to_poses)
            closest_pose = poses[closest_pose_idx]

            for pose in poses:
                if pose.name in self.goal_positions:
                    continue

                target_node_name = f"robot_at_{closest_pose.name}"
                target_node = self.dtg.get(target_node_name)
                target_node = cast(Node, target_node)
                target_nodes.append(target_node)
        else:
            obj_in_hand = self.robot.holding
            obj_in_hand = cast(Object, obj_in_hand)
            goal_pose = obj_in_hand.goal
            goal_pose = cast(Pose, goal_pose)

            target_node_name = f"robot_at_{goal_pose.name}"
            target_node = self.dtg.get(target_node_name)
            target_node = cast(Node, target_node)
            target_nodes.append(target_node)

        return target_nodes

    def evaluate_branches(self, branches: List[Tuple[Node, str, Node]], heuristic: str = 'greedy') -> List[Tuple[Node, str, Node, float]]:
        """
            Evaluate the given branches and assign costs to each branch.
            Return a list of branches with their associated costs.
        """
        evaluated_branches = []

        if len(branches) == 1:
            branch = branches[0]
            branch = (*branch, 0.0)
            return [branch]

        for branch in branches:
            node, action_name, target_node = branch
            action_params = parse_action_params(action_name, node, target_node)

            action_tuple = self.domain.actions.get(action_name)
            action_tuple = cast(Tuple, action_tuple)
            _, conds, _ = action_tuple

            action_applicable = is_action_applicable(conds, action_params)

            if not action_applicable:
                continue

            if heuristic == 'greedy':
                current_pos = self.robot.at.pos
                target_pos = target_node.values[-1].pos
                cost = np.linalg.norm(np.array(current_pos) - np.array(target_pos))

            evaluated_branches.append((node, action_name, target_node, cost))

        return evaluated_branches

    def retrace_action_sequence_back_to_root(self) -> List[List[Action]]:
        action_sequence = []
        action_sequences = []

        for state in self.goal_linked_states:
            while state.parent is not None:
                action = state.parent[0]
                action_sequence.insert(0, action)
                state = state.parent[1]
            action_sequences.append(action_sequence)
            action_sequence = []

        return action_sequences

    def backtrack(self):
        while (not self.current_linked_state.branches_to_explore) or (self.current_linked_state.type_ == StateStatus.GOAL):
            if self.current_linked_state.parent is None:
                print(f"Explored all branches from the root state. Total states explored: {self.state_counter}.")
                self.domain.update_state(self.current_linked_state.state)
                print(self.domain.current_state)
                break

            if self.verbosity == verbose_levels.DEBUG:
                print(f"Back track from {self.current_linked_state.state_id} to {self.current_linked_state.parent[1].state_id}")

            self.current_linked_state = self.current_linked_state.parent[1]
            self.domain.update_state(self.current_linked_state.state)
            self.steps -= 1

    def expand_state(self, s_new: State, action: Action, block_pos: List[str]) -> LinkedState:
        s_new_linked = LinkedState(self.state_counter, s_new, parent=(action, self.current_linked_state))
        self.current_linked_state.weighted_edges.append((action[0], s_new_linked, 0.0))

        self.domain.update_state(s_new)
        self.current_linked_state = s_new_linked
        if self.domain.goal_reached:
            self.current_linked_state.type_ = StateStatus.GOAL
            self.goal_linked_states.append(s_new_linked)
            print(f"Goal reached at state id {s_new_linked.state_id}!, total goal states found: {len(self.goal_linked_states)}",
                  f"steps taken: {self.steps}, num goal blocks: {len(self.goal_blocks)}")
        else:
            self.domain_expansion(block_pos)

        return self.current_linked_state

    def domain_expansion(self, block_pos: List[str]):
        current_nodes = query_available_nodes(self.dtg, self.current_linked_state.state)
        current_nodes = self.prune_unrelated_nodes(current_nodes)

        preferred_action = self.find_preferred_action(block_pos, current_nodes)
        action = self.find_home_and_target_nodes(self.robot.at.name, preferred_action, current_nodes)
        possible_actions = [(action[0], preferred_action, action[1])]
        weighted_actions = self.evaluate_branches(possible_actions, heuristic='greedy')

        self.current_linked_state.branches_to_explore = weighted_actions

    def parse_action_from_branch(self, branch: Tuple[Node, str, Node]) -> Tuple[str, Dict, List[Condition], List[Effect], bool]:
        node, action_name, target_node = branch
        action_params = parse_action_params(action_name, node, target_node)

        action_tuple = self.domain.actions.get(action_name)
        action_tuple = cast(Tuple, action_tuple)
        _, conds, effects = action_tuple

        action_applicable = is_action_applicable(conds, action_params)

        return action_name, action_params, conds, effects, action_applicable

    def find_block_positions(self) -> List[str]:
        block_pos = [cast(Object, obj).at for obj in self.domain.things.get(Object, [])]
        block_pos = [cast(Pose, pos).name for pos in block_pos if pos is not None]
        return block_pos

    def find_preferred_action(self, block_pos: List[str], nodes: List[Node]) -> str:
        """
            Find the the best action to take based on the current state and goal nodes.
        """
        robot_pos = self.robot.at.name
        if robot_pos in block_pos and self.robot.gripper_empty and block_pos not in self.goal_positions:
            action = 'pick'
        elif robot_pos in self.goal_positions and not self.robot.gripper_empty:
            action = 'place'
        else:
            action = 'move'

        return action

    def find_home_and_target_nodes(self, robot_pos: str, action: str, nodes: List[Node]) -> Tuple[Node, Node]:
        match action:
            case 'pick':
                pos = self.domain.name_things.get(robot_pos)
                pos = cast(Pose, pos)

                block_to_pick = pos.occupied_by
                block_to_pick = cast(Object, block_to_pick)

                node_name = f"{block_to_pick.name}_at_{robot_pos}"                
                home_node = self.dtg.get(node_name)
                home_node = cast(Node, home_node)

                target_node_name = f"{block_to_pick.name}_at_None"
                target_node = self.dtg.get(target_node_name)
                target_node = cast(Node, target_node)

            case 'place':
                block_to_place = self.robot.holding
                block_to_place = cast(Object, block_to_place)
                goal_pose = block_to_place.goal
                goal_pose = cast(Pose, goal_pose)

                home_node_name = f"{block_to_place.name}_at_None"
                home_node = self.dtg.get(home_node_name)
                home_node = cast(Node, home_node)

                target_node_name = f"{block_to_place.name}_at_{goal_pose.name}"
                target_node = self.dtg.get(target_node_name)
                target_node = cast(Node, target_node)

            case 'move':
                target_node = self.find_next_move_node(nodes)

                home_node_name = f"robot_at_{self.robot.at.name}"
                home_node = self.dtg.get(home_node_name)
                home_node = cast(Node, home_node)

            case _:
                raise ValueError(f"Unknown action: {action}")

        return home_node, target_node

    def find_next_move_node(self, nodes: List[Node]) -> Node:
        if self.robot.gripper_empty:
            poses = [node.values[2] for node in nodes if node.name.startswith('block')]
            positions = [pose.pos for pose in poses]

            robot_position = self.robot.at.pos
            dists_to_poses = np.linalg.norm(np.array(positions) - np.array(robot_position), axis=1)
            closest_pose_idx = np.argmin(dists_to_poses)
            closest_pose = poses[closest_pose_idx]

            target_node_name = f"robot_at_{closest_pose.name}"
            target_node = self.dtg.get(target_node_name)
            target_node = cast(Node, target_node)
        else:
            obj_in_hand = self.robot.holding
            obj_in_hand = cast(Object, obj_in_hand)
            goal_pose = obj_in_hand.goal
            goal_pose = cast(Pose, goal_pose)

            target_node_name = f"robot_at_{goal_pose.name}"
            target_node = self.dtg.get(target_node_name)
            target_node = cast(Node, target_node)

        return target_node

    def prune_unrelated_nodes(self, nodes: List[Node]) -> List[Node]:
        """
            Node is unrelated if:
                - It is a block node whose
                    - not related to the goal or
                    - it is already at its goal position
        """
        related_nodes = []

        for node in nodes:
            split_node_name = node.name.split('_')
            obj = split_node_name[0]
            pos = split_node_name[-1]

            if obj != 'robot':
                if obj not in self.goal_blocks or pos in self.goal_positions:
                    continue

            related_nodes.append(node)

        return related_nodes

    def log(self, action_name: str, branch: Tuple[Node, str, Node], action_applicable: bool) -> None:
        if self.verbosity == verbose_levels.TRACK:
            self.print_tree(self.s0, self.current_linked_state)

        elif self.verbosity == verbose_levels.DEBUG:
            print(f"Branching: {branch[0].name, branch[1], branch[2].name}")
            print(f"Current state id: {self.current_linked_state.state_id}")

            if action_applicable:
                print(f"Applying action: {action_name} from {branch[0].name} to {branch[2].name}")
            else:
                print(f"Action [{action_name}] not applicable")

    @staticmethod
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
            node_status = "" if node.type_ == StateStatus.ALIVE else f" [{node.type_.name}]"
            print(f"S{node.state_id}{node_status}{node_mark}")

            for action, child, _ in node.weighted_edges:
                child_mark = " <== current" if current is not None and child is current else ""
                child_status = "" if child.type_ == StateStatus.ALIVE else f" [{child.type_.name}]"
                print(f"  └─[{action}]→ S{child.state_id}{child_status}{child_mark}")
                q.append(child)