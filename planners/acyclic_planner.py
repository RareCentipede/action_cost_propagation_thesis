from collections import deque
from enum import Enum
import numpy as np

from eas.block_domain import Pose, Robot, Object, create_goal_nodes
from eas.EAS import Action, apply_action, parse_action_params, is_action_applicable, query_nodes
from eas.EAS import State, Node, Domain, LinkedState, state_state, Condition
from typing import Tuple, Dict, cast, List

verbose_levels = Enum('VerboseLevel', 'NONE DEBUG TRACK INFO')

class AcyclicPlanner:
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

    def run_acyclic_planner(self) -> List[LinkedState]:
        block_pos = self.find_block_positions()
        self.domain_expansion(block_pos)

        shortest_num_steps = np.inf

        while self.current_linked_state.branches_to_explore:
            print(f"{len(self.current_linked_state.branches_to_explore)} branches to explore from state {self.current_linked_state.state_id}.")
            branching = False
            block_pos = self.find_block_positions()
            current_state = self.current_linked_state.state
            branch = self.current_linked_state.branches_to_explore.pop(0)

            action_name, action_params, conds, effects, action_applicable = self.parse_action_from_branch(branch)
            action_args = []
            for param in action_params.values():
                action_args.append(param.name)
            action = Action((action_name, action_args))

            self.log(action_name, branch, action_applicable)

            if action_applicable:
                s_new = apply_action(current_state, conds, action_params, effects)
                branching = self.is_branching_condition_met(s_new, action_name)

            if branching:
                self.state_counter += 1
                self.steps += 1
                self.current_linked_state = self.branch_out(s_new, action, block_pos)
                if self.current_linked_state.type_ == state_state.GOAL:
                    shortest_num_steps = min(self.steps, shortest_num_steps)

            if self.steps >= shortest_num_steps:
                if self.verbosity != verbose_levels.NONE:
                    print("Current path not better than current shortest path, go back to root.")

                self.current_linked_state = self.s0
                self.steps = 0
                self.domain.update_state(self.current_linked_state.state)
            elif (not self.current_linked_state.branches_to_explore):
                if self.verbosity != verbose_levels.NONE:
                    print("No branches to explore, backtracking.")

                self.backtrack()

            if self.verbosity != verbose_levels.NONE:
                print("------------------------------------")

        return self.goal_linked_states

    def retrace_action_sequence_back_to_root(self) -> List[Action]:
        action_sequence = []

        for state in self.goal_linked_states:
            while state.parent is not None:
                action = state.parent[0]
                action_sequence.insert(0, action)
                state = state.parent[1]

        return action_sequence

    def backtrack(self):
        while (not self.current_linked_state.branches_to_explore) or (self.current_linked_state.type_ == state_state.GOAL):
            if self.current_linked_state.parent is None:
                print(f"Explored all branches from the root state. Total states explored: {self.state_counter}.")
                self.domain.update_state(self.current_linked_state.state)
                break

            if self.verbosity == verbose_levels.DEBUG:
                print(f"Back track from {self.current_linked_state.state_id} to {self.current_linked_state.parent[1].state_id}")

            self.current_linked_state = self.current_linked_state.parent[1]
            self.domain.update_state(self.current_linked_state.state)
            self.steps -= 1

    def branch_out(self, s_new: State, action: Action, block_pos: List[str]) -> LinkedState:
        s_new_linked = LinkedState(self.state_counter, s_new, parent=(action, self.current_linked_state))
        self.current_linked_state.edges.append((action[0], s_new_linked))

        self.domain.update_state(s_new)
        self.current_linked_state = s_new_linked
        if self.domain.goal_reached:
            self.current_linked_state.type_ = state_state.GOAL
            self.goal_linked_states.append(s_new_linked)
            print(f"Goal reached at state id {s_new_linked.state_id}!")
        else:
            self.domain_expansion(block_pos)

        return self.current_linked_state

    def domain_expansion(self, block_pos: List[str]):
        current_nodes = query_nodes(self.dtg, self.current_linked_state.state)
        current_nodes = self.prune_unrelated_nodes(current_nodes)
        possible_actions = self.unpack_actions_from_nodes(current_nodes, block_pos)
        self.current_linked_state.branches_to_explore = possible_actions

    def is_branching_condition_met(self, s_new: State, action_name: str) -> bool:
        ancestor = self.current_linked_state.parent
        if ancestor:
            ancestor = ancestor[1]
            if s_new == ancestor.state:
                if self.verbosity == verbose_levels.DEBUG:
                    print("New state is the same as an ancestor state, skipping to avoid cycle.")
                branching = False
            elif (action_name, self.current_linked_state) in ancestor.edges:
                if self.verbosity == verbose_levels.DEBUG:
                    print("Action leading to the new state has already been explored from an ancestor state, skipping to avoid cycle.")
                branching = False
            else:
                branching = True
        else:
            branching = True

        return branching

    def parse_action_from_branch(self, branch: Tuple[Node, str, Node]) -> Tuple[str, Dict, List[Condition], List[Condition], bool]:
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

    def unpack_actions_from_nodes(self, nodes: List[Node], block_pos: List[str]) -> List[Tuple[Node, str, Node]]:
        possible_actions = []

        for node in nodes:
            for edge in node.edges:
                action_name, target_node = edge

                split_base_node_name = node.name.split('_')
                split_target_node_name = target_node.name.split('_')
                base = split_base_node_name[0]
                base_pos = split_base_node_name[-1]

                target_pos = split_target_node_name[-1]

                if base == 'robot' and (target_pos not in block_pos and target_pos not in self.goal_positions):
                    continue
                elif base != 'robot' and ((target_pos == 'None' and base_pos != self.robot.at.name) or \
                    (target_pos != 'None' and target_pos not in self.goal_positions)):
                    continue

                possible_actions.append((node, action_name, target_node))

        return possible_actions

    def prune_unrelated_nodes(self, nodes: List[Node]) -> List[Node]:
        for node in nodes:
            split_node_name = node.name.split('_')
            block = split_node_name[0]
            pos = split_node_name[-1]

            if block != 'robot' and ((pos != self.robot.at.name) and (pos != 'None') or block not in self.goal_blocks):
                nodes.remove(node)

        return nodes

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
            node_status = "" if node.type_ == state_state.ALIVE else f" [{node.type_.name}]"
            print(f"S{node.state_id}{node_status}{node_mark}")

            for action, child in node.edges:
                child_mark = " <== current" if current is not None and child is current else ""
                child_status = "" if child.type_ == state_state.ALIVE else f" [{child.type_.name}]"
                print(f"  └─[{action}]→ S{child.state_id}{child_status}{child_mark}")
                q.append(child)