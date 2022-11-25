from labyrinth_with_deap import compute_subpath, random_individual
from labyrinth_with_deap import creator
import numpy as np

from mazegenerator import MazeGenerator

import time

def random_maze(w : int, h : int, wall_ratio : float) -> np.ndarray:
    maze = np.zeros((h, w), dtype=int)
    for i in range(0, h):
        for j in range(0, w):
            if np.random.rand() < wall_ratio:
                maze[i, j] = 1
    return maze


if __name__ == "__main__":
    # file = "grid10.npy"
    # grid = np.load(file)

    # ind = creator.Individual([3, 2, 1, 1, 1, 3, 1, 1, 3, 2, 3, 3, 3, 3, 3, 1, 1, 0, 1, 0, 1, 3, 3, 0, 3, 1, 1, 1, 3, 3, 1, 1, 2, 2, 1, 0, 2, 2, 3, 3])
    # print(ind)
    # path = compute_subpath(ind, (0, 0), grid, (9, 9))
    # print(path)
    
    # ind = random_individual(int(30 ** 2 / 2))
    
    # file = "realGrid30.npy"
    # grid = np.load(file)
    # start = (1, 1)
    # h = grid.shape[0]
    # w = grid.shape[1]
    # end = (h - 2, w - 2)
        
    # best_path = compute_subpath(ind, start, grid, end)
    
    dim_x, dim_y = 30, 30
    wall_ratio = 0.3
    
    np.save("realisticGrid30.npy", MazeGenerator.generate_maze(dim_x, dim_y))
    