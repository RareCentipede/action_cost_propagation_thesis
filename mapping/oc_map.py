import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from typing import List, Tuple, Dict, cast
from scipy.spatial import KDTree

from eas.EAS import Domain
from eas.block_domain import Pose, Object, Robot

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

        if self.grid_limits is None:
            self.grid_limits = self.compute_grid_limits()

    def compute_grid_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xs = [pose.pos[0] for pose in self.poses]
        ys = [pose.pos[1] for pose in self.poses]

        min_x, max_x = min(xs) - 3*self.col_margin, max(xs) + 3*self.col_margin
        min_y, max_y = min(ys) - 3*self.col_margin, max(ys) + 3*self.col_margin

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

    def create_occupancy_grid_map(self) -> np.ndarray:
        grid = self.create_grid()
        oc_grid = self.assign_occupancy(grid)

        self.oc_grid_map = oc_grid
        return self.oc_grid_map

    def plot_occupancy_grid_map(self, grid: np.ndarray, oc_grid: np.ndarray) -> None:
        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot plot grid.")

        plt.figure(figsize=(8, 8))

        (min_x, max_x), (min_y, max_y) = self.grid_limits

        for point, occ in zip(grid, oc_grid):
            color = 'black' if occ == 1 else 'white'
            rect = patches.Rectangle((point[0], point[1]), self.grid_res, self.grid_res, linewidth=0.5, edgecolor='gray', facecolor=color)
            plt.gca().add_patch(rect)

        for obj in self.objects:
            obj_pos = cast(Pose, obj.at).pos[:2]
            plt.plot(obj_pos[0] + self.grid_res/2, obj_pos[1] + self.grid_res/2, marker='s', color='red', markersize=10)

        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Occupancy Grid Map")
        plt.xlabel("X")
        plt.ylabel("Y")