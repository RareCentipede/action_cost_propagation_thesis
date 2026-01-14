import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from matplotlib.axes import Axes
from typing import List, Tuple, Dict, cast
from scipy.spatial import KDTree

from eas.EAS import Domain, State
from eas.block_domain import Pose, Object

class OccupancyGridMap:
    def __init__(self, domain: Domain, grid_res: float = 0.1, col_margin: float = 1.0, grid_limits: Tuple[Tuple[float, float], Tuple[float, float]] | None = None) -> None:
        self.domain = domain
        self.grid_res = grid_res
        self.col_margin = col_margin
        self.grid_limits = grid_limits
        self.grid_size = None

        self.grid = np.array([])
        self.oc_grid = np.array([])

        self.poses = cast(List[Pose], domain.things.get(Pose))
        self.objects = cast(List[Object], domain.things.get(Object))
        self.obj_positions = [cast(Pose, obj.at).pos for obj in self.objects]
        self.goal_positions = [cast(Pose, obj.goal).pos for obj in self.objects if obj.goal is not None]

        if self.grid_limits is None:
            self.grid_limits = self.compute_grid_limits()

        self.grid_limits = cast(Tuple[Tuple[float, float], Tuple[float, float]], self.grid_limits)

    def compute_grid_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xs = [pose.pos[0] for pose in self.poses]
        ys = [pose.pos[1] for pose in self.poses]

        min_x, max_x = min(xs) - 2/self.grid_res, max(xs) + 2/self.grid_res
        min_y, max_y = min(ys) - 2/self.grid_res, max(ys) + 2/self.grid_res

        grid_limits = ((min_x, max_x), (min_y, max_y))
        return grid_limits

    def create_grid(self) -> np.ndarray:
        grid = np.array([])
        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot create grid.")

        (min_x, max_x), (min_y, max_y) = self.grid_limits

        grid_x = np.arange(min_x, max_x, self.grid_res)
        grid_y = np.arange(min_y, max_y, self.grid_res)

        grid = np.meshgrid(
            grid_x,
            grid_y
        )

        grid = np.array(grid).T.reshape(-1, 2)
        self.grid = grid

        return self.grid

    def assign_occupancy(self, grid: np.ndarray) -> np.ndarray:
        grid_tree = KDTree(grid)
        oc_grid = np.zeros_like(grid[:,0], dtype=int) # 0: free, 1: occupied

        for obj in self.objects:
            obj_pos = cast(Pose, obj.at).pos[:2]
            occupied_indices = grid_tree.query_ball_point(obj_pos, r=self.col_margin)
            oc_grid[occupied_indices] = 1

        self.oc_grid = oc_grid
        return self.oc_grid

    def assign_occupancy_from_state(self, state: State) -> np.ndarray:
        oc_grid = np.zeros_like(self.grid[:,0], dtype=int) # 0: free, 1: occupied
        grid_tree = KDTree(self.grid)

        blocks = self.domain.things.get(Object, [])
        blocks = cast(List[Object], blocks)

        block_names = [f"{block.name}" for block in blocks]
        block_pose_names = [state.get(f"{block_name}_at") for block_name in block_names]
        block_positions = []

        for pose_name in block_pose_names:
            if not pose_name:
                continue

            pose = self.domain.name_things.get(pose_name)
            pose = cast(Pose, pose)
            block_pos = pose.pos[:2]
            block_positions.append(block_pos)
            occupied_indices = grid_tree.query_ball_point(block_pos, r=self.col_margin)
            oc_grid[occupied_indices] = 1

        self.oc_grid = oc_grid
        self.obj_positions = block_positions

        return self.oc_grid

    def create_occupancy_grid_map(self) -> np.ndarray:
        grid = self.create_grid()
        oc_grid = self.assign_occupancy(grid)

        self.oc_grid_map = oc_grid
        return self.oc_grid_map

    def plot_occupancy_grid_map(self, grid: np.ndarray, oc_grid: np.ndarray, ax: Axes | None = None):
        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot plot grid.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        (min_x, max_x), (min_y, max_y) = self.grid_limits

        for point, occ in zip(grid, oc_grid):
            color = 'black' if occ == 1 else 'white'
            rect = patches.Rectangle((point[0]-self.grid_res/2, point[1]-self.grid_res/2),
                                     self.grid_res, self.grid_res, linewidth=0.5, edgecolor='gray', facecolor=color)
            ax.add_patch(rect)

        for obj_pos in self.obj_positions:
            rect = patches.Rectangle((obj_pos[0]-self.grid_res/2, obj_pos[1]-self.grid_res/2),
                                     self.grid_res, self.grid_res, linewidth=0.5, edgecolor='gray', facecolor='red')
            ax.add_patch(rect)

        for goal_pos in self.goal_positions:
            ax.scatter(goal_pos[0], goal_pos[1], s=100, c='green', marker='*')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Occupancy Grid Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")