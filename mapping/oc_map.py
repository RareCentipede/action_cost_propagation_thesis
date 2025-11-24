import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, cast
from eas.EAS import Domain
from eas.block_domain import Pose, Object, Robot
from matplotlib import patches

class OccupancyGridMap:
    def __init__(self, domain: Domain, grid_res: float = 0.1, col_margin: float = 1.0, grid_limits: Tuple[Tuple[float, float], Tuple[float, float]] | None = None) -> None:
        self.domain = domain
        self.grid_res = grid_res
        self.col_margin = col_margin
        self.grid_limits = grid_limits
        self.grid_size = None

        self.poses = cast(List[Pose], domain.things.get(Pose))
        self.objects = cast(List[Object], domain.things.get(Object))

        if self.grid_limits is None:
            self.grid_limits = self.compute_grid_limits()

        self.grid_size = self.compute_grid_size()

    def compute_grid_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xs = [pose.pos[0] for pose in self.poses]
        ys = [pose.pos[1] for pose in self.poses]

        min_x, max_x = min(xs) - self.grid_res, max(xs) + self.grid_res
        min_y, max_y = min(ys) - self.grid_res, max(ys) + self.grid_res

        grid_limits = ((min_x, max_x), (min_y, max_y))
        return grid_limits

    def compute_grid_size(self) -> Tuple[int, int]:
        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot compute grid size.")

        (min_x, max_x), (min_y, max_y) = self.grid_limits

        x_size = int(np.ceil((max_x - min_x) / self.grid_res))
        y_size = int(np.ceil((max_y - min_y) / self.grid_res))

        self.grid_size = (x_size, y_size)
        return self.grid_size

    def create_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.array([])
        if self.grid_size is None:
            raise ValueError("Grid size not computed. Cannot create grid.")

        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot create grid.")

        (min_x, max_x), (min_y, max_y) = self.grid_limits
        x_size, y_size = self.grid_size

        grid = np.meshgrid(
            np.linspace(min_x, max_x, x_size),
            np.linspace(min_y, max_y, y_size)
        )

        return grid

    def create_occupancy_grid_map(self) -> np.ndarray:
        grid = np.array([])
        if self.grid_size is None:
            raise ValueError("Grid size not computed. Cannot create grid.")

        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot create grid.")

        (min_x, max_x), (min_y, max_y) = self.grid_limits
        x_size, y_size = self.grid_size

        grid = np.zeros((x_size, y_size), dtype=int)

        for obj in self.objects:
            obj_pose = cast(Pose, obj.at)
            obj_x, obj_y, _ = obj_pose.pos
            obj_x, obj_y = abs(int(obj_x - min_x)), abs(int(max_y - obj_y))

            print(f"Object {obj.name} of cartesian position ({obj_pose.pos[0]}, {obj_pose.pos[1]}) at grid position ({obj_x}, {obj_y})")

            col_radius = int(np.ceil(self.col_margin))
            occupied_extents = [np.arange(max(0, obj_x - col_radius), min(x_size, obj_x + col_radius + 1)),
                               np.arange(max(0, obj_y - col_radius), min(y_size, obj_y + col_radius + 1))]
            grid[np.ix_(occupied_extents[0], occupied_extents[1])] = 1

        return grid

    def plot_occupancy_grid_map(self, grid: np.ndarray) -> None:
        if self.grid_limits is None:
            raise ValueError("Grid limits not set. Cannot plot grid.")

        if self.grid_size is None:
            raise ValueError("Grid size not computed. Cannot plot grid.")

        plt.figure(figsize=(8, 8))

        # Plot black dots for occupied cells
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1:
                    rect = patches.Rectangle((i * self.grid_res, j * self.grid_res), self.grid_res, self.grid_res, color='black')
                    plt.gca().add_patch(rect)

        # Make grid the same resolution as specified
        # plt.xticks(np.arange(self.grid_limits[0][0], self.grid_limits[0][1], self.grid_res))
        # plt.yticks(np.arange(self.grid_limits[1][0], self.grid_limits[1][1], self.grid_res))
        plt.grid(True)

        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('Occupancy Grid Map')
        plt.xlim(self.grid_limits[0][0], self.grid_limits[0][1])
        plt.ylim(self.grid_limits[1][0], self.grid_limits[1][1])
        plt.show()