import multiprocessing as mp
from labyrinth_with_deap import solve_labyrinth
import numpy as np

if __name__ == "__main__":
    file = "grid10.npy"
    grid = np.load(file)
    start = (0, 0)
    h = grid.shape[0]
    w = grid.shape[1]
    end = (h - 1, w - 1)

    pool = mp.Pool(mp.cpu_count())

    print(mp.cpu_count())

    results = pool.starmap(solve_labyrinth, [(grid, start, end, 3) for i in range(100)])

    lengths = [len(path) for path in results]

    pool.close()
        
    print(lengths)