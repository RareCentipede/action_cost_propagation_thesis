import numpy as np

from collections import deque
from enum import Enum
from typing import Tuple, Dict, cast, List

from eas.block_domain import Pose, Robot, Object, create_goal_nodes, define_neighbor_preferences
from eas.EAS import Action, Effect, apply_action, parse_action_params, is_action_applicable, query_available_nodes, query_nodes
from eas.EAS import State, Node, Domain, LinkedState, StateStatus, Condition
from mapping.oc_map import OccupancyGridMap
from mapping.path_planner import create_nx_nodes, astar
from cost_propagation.cost_propagation import spawn_blocks, perform_cost_propagation, visualize_cost_propagation

verbose_levels = Enum('VerboseLevel', 'NONE DEBUG TRACK INFO')
heuristic_types = Enum('HeuristicType', 'NONE LAZY_GREEDY LAZY_GREEDY_PROPAGATED \
                        DILIGENT_GREEDY DILIGENT_GREEDY_PROPAGATED \
                        GREEDY_NEIGHBOR GREEDY_NEIGHBOR_PROPAGATED')

class OrderedLandmarksPlanner:
    def __init__(self, domain: Domain, dtg: Dict[str, Node], ocm: OccupancyGridMap,
                 verbosity: verbose_levels = verbose_levels.NONE):
        self.domain = domain
        self.dtg = dtg
        self.ocm = ocm
        self.verbosity = verbosity
        self.greedy = False

        self.goal_nodes = create_goal_nodes(self.domain, self.dtg)
        self.current_state = self.domain.current_state
        self.goal_blocks = [g_node.values[1].name for g_node in self.goal_nodes.values()]
        self.goal_positions = [g_node.values[-1].name for g_node in self.goal_nodes.values()]

        self.state_counter = 0
        self.steps = 0
        self.s0 = LinkedState(state=self.current_state, state_id=self.state_counter)
        self.current_linked_state = self.s0
        self.goal_linked_states = []
        self.theoretical_min_steps = len(self.goal_blocks) * 2  # Each block requires at least a pick and place
        self.moves = 0

        self.propagated_costs_dict = {}

        robot = domain.things.get(Robot, [])[0]
        self.robot = cast(Robot, robot)

    def run_greedy_ordered_landmarks_planner(self, heuristic: heuristic_types = heuristic_types.LAZY_GREEDY) -> LinkedState | None:
        self.greedy = True
        min_cost = np.inf

        nodes = query_nodes(self.dtg, self.domain.current_state)
        nodes = self.prune_unrelated_nodes(nodes)

        block_nodes = [node for node in nodes if not node.name.startswith('robot')]
        goal_nodes = [node for node in self.goal_nodes.values()]
        define_neighbor_preferences(block_nodes, goal_nodes)

        # Define initial branches at root state
        goal_linked_state, cost_to_goal = self.search_for_path_from_state_to_goal(self.s0, heuristic, min_cost)
        if goal_linked_state:
            print(f"Found goal linked state at cost: {cost_to_goal}, steps taken: {self.steps}")
        else:
            print(f"Could not reach goal from root state :(")

        return goal_linked_state

    def run_optimal_ordered_landmarks_planner(self, heuristic: heuristic_types = heuristic_types.LAZY_GREEDY) -> Tuple[List[LinkedState], LinkedState]:
        goal_linked_states = []
        self.greedy = False
        best_goal_linked_state = self.s0
        shortest_num_steps = np.inf
        min_cost = np.inf
        current_cost = 0.0

        nodes = query_nodes(self.dtg, self.domain.current_state)
        nodes = self.prune_unrelated_nodes(nodes)

        block_nodes = [node for node in nodes if not node.name.startswith('robot')]
        goal_nodes = [node for node in self.goal_nodes.values()]
        define_neighbor_preferences(block_nodes, goal_nodes)

        # Define initial branches at root state
        weighted_branches = self.branch_out(self.s0, heuristic)
        self.s0.branches_to_explore = weighted_branches
        self.s0.branches_to_explore.append(self.s0.branches_to_explore[0])

        # Explore each branch until goal reached in each one
        # If goal cannot be reached at a branch, skip it.
        # Choose the cheapest branch as the final plan.
        while self.s0.branches_to_explore:
            print(f"Starting new exploration from root state, remaining branches: {len(self.s0.branches_to_explore)}")
            self.current_linked_state = self.s0
            weighted_branch = self.current_linked_state.branches_to_explore.pop(0)
            current_state = self.current_linked_state.state

            self.domain.update_state(current_state)
            self.steps = 0

            new_linked_state = self.expand_from_branch(self.current_linked_state, weighted_branch)
            if not new_linked_state:
                print(f"Could not expand from branch {weighted_branch} at root, skipping.")
                continue

            self.current_linked_state = new_linked_state
            current_cost = self.current_linked_state.cost
            self.steps += 1

            goal_linked_state, cost_to_goal = self.search_for_path_from_state_to_goal(self.current_linked_state, heuristic, min_cost)
            current_cost += cost_to_goal

            if goal_linked_state:
                goal_linked_states.append(goal_linked_state)
                print(f"Found goal linked state at cost: {current_cost}, steps taken: {self.steps}")

                if self.steps < shortest_num_steps:
                    shortest_num_steps = self.steps

                if current_cost < min_cost:
                    print(f"Minimum cost updated from {min_cost} to {current_cost}, at steps: {self.steps}")
                    min_cost = current_cost
                    best_goal_linked_state = goal_linked_state
            else:
                print(f"Could not reach goal from current branch at root, moving to next branch.")

        return goal_linked_states, best_goal_linked_state

    def search_for_path_from_state_to_goal(self, start_linked_state: LinkedState, heuristic: heuristic_types, min_cost: float) -> Tuple[LinkedState | None, float]:
        goal_linked_states = None
        current_linked_state = start_linked_state
        current_cost = current_linked_state.cost

        while not self.domain.goal_reached and ((self.greedy and (current_linked_state.state_id >= 0)) or \
        (not self.greedy and (current_linked_state.state_id > 0))):
            self.current_linked_state = current_linked_state
            current_cost = current_linked_state.cost

            weighted_branches = self.branch_out(current_linked_state, heuristic)
            if not weighted_branches:
                print(f"No more branches to explore from state id {current_linked_state.state_id}, backtracking.")
                current_linked_state = self.backtrack(0)
                continue

            if self.verbosity == verbose_levels.INFO:
                print(f"Exploring from state id {current_linked_state.state_id}, steps taken: {self.steps}, current cost: {current_cost}")

            current_linked_state.branches_to_explore = weighted_branches
            weighted_selected_branch = current_linked_state.branches_to_explore.pop(0)

            selected_branch = weighted_selected_branch[:-1]
            current_cost += weighted_selected_branch[3]

            if self.verbosity == verbose_levels.DEBUG:
                print(f"Selected branch to expand: {selected_branch[0].name} --[{selected_branch[1]}]--> {selected_branch[2].name}")

            new_linked_state = self.expand_from_branch(current_linked_state, weighted_selected_branch)
            if not new_linked_state or (current_cost >= min_cost):
                if self.verbosity == verbose_levels.DEBUG:
                    if not new_linked_state:
                        print(f"Could not expand from branch at state id {current_linked_state.state_id}.")
                    else:
                        print(f"Current cost {current_cost} exceeded min cost {min_cost}.")

                if len(current_linked_state.branches_to_explore) > 0:
                    print("Trying next branch from current state.")
                    continue

                current_linked_state = self.backtrack(0)
                continue

            current_linked_state = new_linked_state

            if self.domain.goal_reached:
                goal_linked_states = current_linked_state

        return goal_linked_states, current_cost

    def expand_from_branch(self, current_linked_state: LinkedState, weighted_branch: Tuple[Node, str, Node, float]) -> LinkedState | None:
        self.steps += 1
        branch = weighted_branch[:-1]  # Remove cost
        branch_cost = weighted_branch[3]
        cost = current_linked_state.cost

        action_name, action_params, conds, effects, action_applicable = self.parse_action_from_branch(branch)
        action_args = []
        for param in action_params.values():
            action_args.append(param.name)
        action = Action((action_name, action_args))

        self.log(action_name, branch, action_applicable)

        if not action_applicable:
            print(f"Action [{action_name}] not applicable.")
            return None

        s_new = apply_action(current_linked_state.state, conds, action_params, effects)

        if action_name == 'move' and self.robot.gripper_empty:
            cost += branch_cost

        return self.expand_state(current_linked_state, s_new, action, cost)

    def branch_out(self, current_linked_state: LinkedState, heuristic: heuristic_types) -> List[Tuple[Node, str, Node, float]]:
        """
            Find useful branches to explore from the current state. Return the updated linked state with branches to explore.
        """
        if current_linked_state.branches_to_explore:
            if self.verbosity == verbose_levels.DEBUG:
                print(f"Using existing branches to explore from state id {current_linked_state.state_id}, num branches: {len(current_linked_state.branches_to_explore)}")

            return current_linked_state.branches_to_explore

        available_nodes = query_available_nodes(self.dtg, current_linked_state.state)
        available_nodes = self.prune_unrelated_nodes(available_nodes)
        block_pos = self.find_block_positions()
        preferred_action = self.find_preferred_action(block_pos, available_nodes)

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
                target_nodes = self.define_move_branches(available_nodes)
                for target_node in target_nodes:
                    branches.append((home_node, preferred_action, target_node))
            case _:
                raise ValueError(f"Unknown action: {preferred_action}")

        weighted_branches = self.evaluate_branches(branches, heuristic)

        return weighted_branches

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
            If gripper empty, many choices for movement. If gripper already holding an object, only move
            to its goal position.
        """
        target_nodes = []

        if self.robot.gripper_empty:
            poses = [node.values[2] for node in nodes if node.name.startswith('block')]

            for pose in poses:
                if pose.name in self.goal_positions:
                    continue

                target_node_name = f"robot_at_{pose.name}"
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

    def evaluate_branches(self, branches: List[Tuple[Node, str, Node]], heuristic: heuristic_types = heuristic_types.LAZY_GREEDY) -> List[Tuple[Node, str, Node, float]]:
        """
            Evaluate the given branches and assign costs to each branch.
            Return a list of branches with their associated costs.
        """
        evaluated_branches = []
        propagated_heuristics = [heuristic_types.LAZY_GREEDY_PROPAGATED,
                                   heuristic_types.DILIGENT_GREEDY_PROPAGATED,
                                   heuristic_types.GREEDY_NEIGHBOR_PROPAGATED]

        for branch in branches:
            node, action_name, target_node = branch

            if action_name != 'move' or not self.robot.gripper_empty:
                branch = branches[0]
                branch = (*branch, 0.0)
                return [branch]

            action_params = parse_action_params(action_name, node, target_node)

            action_tuple = self.domain.actions.get(action_name)
            action_tuple = cast(Tuple, action_tuple)
            _, conds, _ = action_tuple

            action_applicable = is_action_applicable(conds, action_params)

            if not action_applicable:
                continue

            if heuristic in propagated_heuristics:
                blocks_obj_dict, block_positions = spawn_blocks(self.domain)
                scaled_projected_vecs_lists, self.propagated_cost_dict = perform_cost_propagation(blocks_obj_dict, block_positions)
                # visualize_cost_propagation(blocks_obj_dict, block_positions,
                                        #    self.propagated_cost_dict, scaled_projected_vecs_lists, self.robot.at.pos)

            match heuristic:
                case heuristic_types.LAZY_GREEDY:
                    cost = self.lazy_greedy_heuristic(self.robot.at.pos, target_node)
                case heuristic_types.LAZY_GREEDY_PROPAGATED:
                    cost = self.lazy_greedy_heuristic(self.robot.at.pos, target_node, propagate=True)
                case heuristic_types.DILIGENT_GREEDY:
                    cost = self.diligent_greedy_heuristic(self.robot.at.pos, target_node)
                case heuristic_types.DILIGENT_GREEDY_PROPAGATED:
                    cost = self.diligent_greedy_heuristic(self.robot.at.pos, target_node, propagate=True)
                case heuristic_types.GREEDY_NEIGHBOR:
                    cost = self.greedy_neighbor_heuristic(target_node)
                case heuristic_types.GREEDY_NEIGHBOR_PROPAGATED:
                    cost = self.greedy_neighbor_heuristic(target_node, propagate=True)
                case _:
                    cost = 0.0

            evaluated_branches.append((node, action_name, target_node, cost))
            if self.verbosity == verbose_levels.DEBUG:
                print(f"Evaluated branch: {node.name} --[{action_name}]--> {target_node.name} with cost: {cost}")

        evaluated_branches.sort(key=lambda x: x[3])  # Sort by cost

        return evaluated_branches

    def lazy_greedy_heuristic(self, current_pos: Tuple[float, float, float], target_node: Node, propagate: bool = False) -> float:
        target_pos = target_node.values[-1].pos
        ground_pt = (target_pos[0], target_pos[1], 0.5)
        cost = np.linalg.norm(np.array(current_pos) - np.array(ground_pt))
        cost += np.linalg.norm(np.array(ground_pt) - np.array(target_pos))

        target_obj = target_node.values[-1].occupied_by
        target_obj = cast(Object, target_obj)
        target_obj_goal = target_obj.goal
        target_obj_goal = cast(Pose, target_obj_goal)

        goal_pos = target_obj_goal.pos
        ground_pt = (goal_pos[0], goal_pos[1], 0.5)
        cost += np.linalg.norm(np.array(target_pos) - np.array(ground_pt))
        cost += np.linalg.norm(np.array(ground_pt) - np.array(goal_pos))

        if propagate:
            p_cost = self.propagated_costs_dict.get(target_obj.name, 0.0)
            cost += p_cost

        if self.verbosity == verbose_levels.INFO:
            if propagate:
                print(f"Steps: {self.steps}, Propagated cost: {p_cost:.2f}, Final cost: {cost:.2f}")
            else:
                print(f"Steps: {self.steps}, Final cost: {cost:.2f}")

        return cost.item()

    def diligent_greedy_heuristic(self, current_pos: Tuple[float, float, float], target_node: Node, propagate: bool = False, p_discount_factor: float | None = None) -> float:
        cost = 0.0

        target_obj = target_node.values[-1].occupied_by
        target_obj_pos = target_node.values[-1].pos
        target_obj = cast(Object, target_obj)
        target_obj_goal = target_obj.goal
        target_obj_goal = cast(Pose, target_obj_goal)
        target_goal_pos = target_obj_goal.pos

        self.ocm.assign_occupancy_from_state(self.current_linked_state.state)
        graph = create_nx_nodes(self.ocm)

        start = (current_pos[0], current_pos[1])
        goal = (target_obj_pos[0], target_obj_pos[1])

        path_to_block = np.array(astar(graph, self.ocm.oc_grid, start, goal))
        cost += np.sum(np.linalg.norm(np.diff(path_to_block, axis=0), axis=1))

        start = (target_obj_pos[0], target_obj_pos[1])
        goal = (target_goal_pos[0], target_goal_pos[1])    

        path_block_to_goal = np.array(astar(graph, self.ocm.oc_grid, start, goal))
        cost += np.sum(np.linalg.norm(np.diff(path_block_to_goal, axis=0), axis=1))

        if propagate:
            cost += self.propagated_costs_dict.get(target_obj.name, 0.0)

        return cost.item()

    def greedy_neighbor_heuristic(self, target_node: Node, propagate: bool = False) -> float:
        cost = 0.0
        visited_node_count = 0
        nodes_to_visit = len(self.goal_blocks)

        block_positions = self.find_block_positions()
        blocks_at_goal_positions = [pos for pos in block_positions if pos in self.goal_positions]
        nodes_to_visit -= len(blocks_at_goal_positions)

        block = target_node.values[1].occupied_by
        block = cast(Object, block)

        # First sum up greedy costs for the current target
        current_cost = self.lazy_greedy_heuristic(self.robot.at.pos, target_node)
        cost += current_cost
        visited_node_count += 1

        cost = self.compute_cost_to_preferred_neighbor(visited_node_count, nodes_to_visit, block, propagate)

        return cost

    def compute_cost_to_preferred_neighbor(self, visited_node_count: int, nodes_to_visit: int, initial_block: Object, propagate: bool = False) -> float:
        cost = 0.0
        visited_neighbors = []
        block = initial_block
        ranked_neighbors = block.ranked_neighbors

        while visited_node_count < nodes_to_visit:
            for neighbor_name in ranked_neighbors:
                if (neighbor_name not in visited_neighbors):
                    visited_neighbors.append(neighbor_name)
                    break

            neighbor_obj = self.domain.name_things.get(neighbor_name)
            neighbor_obj = cast(Object, neighbor_obj)
            block.preferred_neighbor = neighbor_name
            # if self.verbosity == verbose_levels.DEBUG:
                # print(f"Block {block.name} ranked neighbors: {ranked_neighbors}, preferred neighbor: {neighbor_name}")

            neighbor_pose = neighbor_obj.at
            neighbor_pose = cast(Pose, neighbor_pose)

            neighbor_node_name = f"{neighbor_obj.name}_at_{neighbor_pose.name}"
            neighbor_node = self.dtg.get(neighbor_node_name)
            neighbor_node = cast(Node, neighbor_node)

            target_goal = block.goal
            target_goal = cast(Pose, target_goal)
            target_goal_pos = target_goal.pos

            # Add distance between target's goal and neighbor to cost
            visited_node_count += 1
            cost += self.lazy_greedy_heuristic(target_goal_pos, neighbor_node, propagate)

            # Set neighbor as the new target and keep adding costs until all goal blocks are visited
            block = neighbor_obj
            ranked_neighbors = block.ranked_neighbors

        return cost

    def expand_state(self, current_linked_state: LinkedState, s_new: State, action: Action, cost: float = 0.0) -> LinkedState:
        self.state_counter += 1

        s_new_linked = LinkedState(self.state_counter, s_new, parent=(action, current_linked_state), cost=cost)
        current_linked_state.weighted_edges.append((action[0], s_new_linked, cost))
        self.domain.update_state(s_new)

        if self.domain.goal_reached:
            s_new_linked.type_ = StateStatus.GOAL
            self.goal_linked_states.append(s_new_linked)
            print(f"Goal reached at state id {s_new_linked.state_id}!, total goal states found: {len(self.goal_linked_states)}",
                  f"steps taken: {self.steps}, num goal blocks: {len(self.goal_blocks)}")

        return s_new_linked

    def retrace_action_sequence_back_to_root(self, goal_linked_state: LinkedState) -> Tuple[List[Action], List[State]]:
        action_sequence = []
        states = []
        state = goal_linked_state

        while state.parent is not None:
            action = state.parent[0]
            action_sequence.insert(0, action)

            states.insert(0, state.state)
            state = state.parent[1]

        return action_sequence, states

    def retrace_optimal_action_sequence_back_to_root(self) -> Tuple[List[Action], List[State]]:
        action_sequences = []
        states_lists = []
        total_sequence_costs = []
        sequence_costs = []

        if len(self.goal_linked_states) == 0:
            print("No goal linked states to retrace.")
            return [], []

        for state in self.goal_linked_states:
            current_sequence_costs = []
            action_sequence = []
            states = []

            while state.parent is not None:
                action = state.parent[0]
                action_sequence.insert(0, action)

                states.insert(0, state.state)
                state = state.parent[1]

                current_sequence_costs.insert(0, state.cost)

            action_sequences.append(action_sequence)
            sequence_costs.append(current_sequence_costs)
            states_lists.append(states)

        for seq_costs in sequence_costs:
            total_cost = 0.0

            for step_cost in seq_costs:
                total_cost += step_cost

            total_sequence_costs.append(total_cost)

        total_sequence_costs = np.array(total_sequence_costs)
        print(f"Total sequence costs for all goal states: {total_sequence_costs}")

        optimal_idx = int(np.argmin(np.array(total_sequence_costs)))
        optimal_actions = action_sequences[optimal_idx]
        optimal_states = states_lists[optimal_idx]

        return optimal_actions, optimal_states

    def backtrack(self, stopping_state_id: int | None = None) -> LinkedState:
        current_linked_state = self.current_linked_state
        while not current_linked_state.branches_to_explore:
            if current_linked_state.parent is None:
                print("Returned to root, terminating search on this branch.")
                self.domain.update_state(current_linked_state.state)
                break

            elif stopping_state_id is not None and current_linked_state.state_id == stopping_state_id:
                print(f"Reached stopping state id {stopping_state_id}, halting backtrack.")
                self.domain.update_state(current_linked_state.state)
                break

            if self.verbosity == verbose_levels.DEBUG:
                print(f"Back track from {current_linked_state.state_id} to {current_linked_state.parent[1].state_id}, branches: {len(current_linked_state.parent[1].branches_to_explore)}, steps: {self.steps}")

            current_linked_state = current_linked_state.parent[1]
            self.domain.update_state(current_linked_state.state)
            self.steps -= 1

        return current_linked_state

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
        if robot_pos in block_pos and self.robot.gripper_empty and robot_pos not in self.goal_positions:
            action = 'pick'
        elif robot_pos in self.goal_positions and not self.robot.gripper_empty:
            action = 'place'
        else:
            action = 'move'

        return action

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
            print(f"Current state id: {self.current_linked_state.state_id+1}")

            if action_applicable:
                print(f"Applying action: {action_name} from {branch[0].name} to {branch[2].name}")

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