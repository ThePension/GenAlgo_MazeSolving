from labyrinth_with_deap import compute_subpath, random_individual
from labyrinth_with_deap import creator
import numpy as np

import time

if __name__ == "__main__":
    file = "grid10.npy"
    grid = np.load(file)

    ind = creator.Individual([3, 2, 1, 1, 1, 3, 1, 1, 3, 2, 3, 3, 3, 3, 3, 1, 1, 0, 1, 0, 1, 3, 3, 0, 3, 1, 1, 1, 3, 3, 1, 1, 2, 2, 1, 0, 2, 2, 3, 3])
    print(ind)
    path = compute_subpath(ind, (0, 0), grid, (9, 9))
    print(path)
    