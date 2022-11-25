import multiprocessing as mp
from labyrinth_with_deap import solve_labyrinth
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file = "./test_grids/realisticGrid30.npy"
    grid = np.load(file)
    grid[1][1] = 0
    start = (0, 0)
    h = grid.shape[0]
    w = grid.shape[1]
    end = (h - 1, w - 1)

    pool = mp.Pool(mp.cpu_count())

    results = pool.starmap(solve_labyrinth, [(grid, start, end, 180) for i in range(100)])

    lengths = [len(path) for path in results]

    pool.close()
    
    # Set the inverval for the bins to 1
    plt.xticks(np.arange(min(lengths), max(lengths)+1, 1.0))
        
    plt.hist(lengths)
    
    # Add labels for length of path
    plt.xlabel("Length of path")
    
    # Add labels for number of paths
    plt.ylabel("Number of paths")
    
    plt.show()