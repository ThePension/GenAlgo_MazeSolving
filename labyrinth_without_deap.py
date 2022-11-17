"""Main class for solving labyrinths with genetic algorithms.

Tested Python 3.9+
"""

# Sauces :
# - https://www.diva-portal.org/smash/get/diva2:927325/FULLTEXT01.pdf

import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
from individual import Individual
from mazegenerator import MazeGenerator

def display_labyrinth(grid:np.ndarray, start_cell:tuple, end_cell:tuple, solution=None):
    """Display the labyrinth matrix and possibly the solution with matplotlib.
    Free cell will be in light gray.
    Wall cells will be in dark gray.
    Start and end cells will be in dark blue.
    Path cells (start, end excluded) will be in light blue.
    :param grid np.ndarray: labyrinth matrix
    :param start_cell: tuple of i, j indices for the start cell
    :param end_cell: tuple of i, j indices for the end cell
    :param solution: list of successive tuple i, j indices who forms the path
    """
    grid = np.array(grid, copy=True)
    FREE_CELL = 19
    WALL_CELL = 16
    START = 0
    END = 0
    PATH = 2
    grid[grid == 0] = FREE_CELL
    grid[grid == 1] = WALL_CELL
    grid[start_cell] = START
    grid[end_cell] = END
    if solution:
        solution = solution[1:-1]
        for cell in solution:
            grid[cell] = PATH
    else:
        print("No solution has been found")
    plt.matshow(grid, cmap="tab20c")
    
def solve_labyrinth(grid:np.ndarray, start_cell:tuple, end_cell:tuple, max_time_s:float) -> list[tuple[int, int]]:
    """Attempt to solve the labyrinth by returning the best path found
    :param grid np.array: numpy 2d array
    :start_cell tuple: tuple of i, j indices for the start cell
    :end_cell tuple: tuple of i, j indices for the end cell
    :max_time_s float: maximum time for running the algorithm
    :return list: list of successive tuple i, j indices who forms the path
    """
    # ------------------------------------------
    #
    # Useful variables
    #
    # ------------------------------------------
    
    start_time = time.time() # Start timer - Used to keep track of the elapsed time

    # Longest path possible is the number of cells in the grid / 2
    h, w = grid.shape
    adn_size = (w + h) * 2
    population_size =  w + h # int((w + h) * (4 / 3))
    mutation_rate = 0.7 # Probability of mutation for a individual
    gene_mutation_rate = 0.1 # Probability of mutation for a gene
    mating_rate = 0.7 # Probability of crossover for a pair of individuals
    ellitiste_mutation_rate = 0.01
    gen_count = 0

    population = [Individual.randomIndividual(start_cell, adn_size, grid) for i in range(0, population_size)]
    best_ind = population[0] # Keep track of the best path found for the moment

    while(True):
        # -----------------------------------------
        #
        # Compute the next generation
        #
        # ------------------------------------------
        ind, new_population = compute_generation(population, grid, end_cell, mutation_rate, mating_rate, gene_mutation_rate, ellitiste_mutation_rate)
        
        gen_count += 1
        
        population[:] = new_population

        ind.compute_fitness(end_cell)

        if (ind.fitness < best_ind.fitness):
            best_ind = ind

        print("NPOP : " + str(len(population)) + " | Generation: " + str(gen_count) + " - Best path fitness: " + str(best_ind.fitness))

        # ------------------------------------------
        #
        # Check if the algorithm has to stop
        #
        # ------------------------------------------
        if time.time() - start_time >= max_time_s:
            break

    return best_ind.extract_partial_path(end_cell)

def compute_generation(population:list[Individual], grid:np.ndarray, target:tuple, mutation_rate : float, mating_rate : float, gene_mutation_rate : float, ellitiste_mutation_rate : float = 0) -> Individual:
    ellitiste_number = round(ellitiste_mutation_rate * len(population))

    bests = population[:ellitiste_number:]
    population = population[ellitiste_number::]
    
    offspring = selection(population, int(len(population) / 2))

    offspring = [ind.clone() for ind in offspring]

    for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
        if randint(0, 100) < mating_rate * 100:
            child1, child2 = crossover(parent1, parent2, target)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1.clone())
            offspring.append(parent2.clone())

    for ind in offspring:
        if randint(0, 100) < mutation_rate * 100:
            ind.mutate(gene_mutation_rate)
            
        ind.compute_fitness(target)

    population = bests + offspring

    # shuffle the population to improve diversity
    np.random.shuffle(population)

    # Get the 5 best individuals
    bestPaths = selection(population, 5)

    [print("Best path fitness in this gen : " + str(ind.fitness)) for ind in bestPaths]

    return bestPaths[0].clone(), population # return the best individual
    

def selection(population : list[Individual], k : int) -> list[Individual]:
    return sorted(population, key=lambda x: x.fitness, reverse=False)[:k]

def crossover(parent1 : Individual, parent2 : Individual, target : tuple[int, int]) -> tuple[Individual, Individual]:
    adn_parent1 = parent1.clone().adn
    adn_parent2 = parent2.clone().adn

    crossing_point = round(len(adn_parent1) / 2)
    
    # V1
    adn_child1 = adn_parent1[crossing_point:] + adn_parent2[:crossing_point]
    adn_child2 = adn_parent1[crossing_point:] + adn_parent1[:crossing_point]
    # END V1
    
    # V2
    # adn_child1 = []
    # adn_child2 = []

    # for i in range(0, len(adn_parent1)):
    #     if randint(0, 1) == 0:
    #         adn_child1.append(adn_parent1[i])
    #         adn_child2.append(adn_parent2[i])
    #     else:
    #         adn_child1.append(adn_parent2[i])
    #         adn_child2.append(adn_parent1[i])
    # END V2

    # V3
    # adn_child1 = []
    # adn_child2 = []
    # pos1 = 0

    # if len(parent1.extract_partial_path(target)) > len(parent2.extract_partial_path(target)):
    #     pos1 = len(parent1.extract_partial_path(target))
    # else:
    #     pos1 = len(parent2.extract_partial_path(target))

    # print("Pos1 : " + str(pos1))

    # pos2 = randint(0, len(adn_parent1) - pos1) + pos1

    # for i in range(0, pos1 - 1):
    #     adn_child1[i] = parent1.adn[i]
    #     adn_child2[i] = parent2.adn[i]

    # for i in range(i, pos2 - 1):
    #     adn_child1[i] = adn_parent2[i]
    #     adn_child2[i] = adn_parent1[i]

    # for i in range(i, len(parent1.adn)):
    #     adn_child1[i] = parent1.adn[i]
    #     adn_child2[i] = parent2.adn[i]
    
    child1 = Individual(parent1.init_cell, adn_child1, parent1.maze)
    child2 = Individual(parent1.init_cell, adn_child2, parent1.maze)

    return child1, child2   

def random_maze(w : int, h : int, wall_ratio : float) -> np.ndarray:
    maze = np.zeros((h, w), dtype=int)
    for i in range(0, h):
        for j in range(0, w):
            if np.random.rand() < wall_ratio:
                maze[i, j] = 1
    return maze

if __name__ == "__main__":
    import numpy as np

    file = "grid10.npy"
    grid = np.load(file)
    start = (0, 0)
    h = grid.shape[0]
    w = grid.shape[1]
    end = (h - 1, w - 1)
        
    best_path = solve_labyrinth(grid, start, end, 3)

    print(best_path)
    
    # size = 15
    # init_cell = (1, 1)
    # dest_cell = (size - 2, size - 2)
    # time_s = 15
    
    # m = MazeGenerator.generate_maze(15, 15)

    # path = solve_labyrinth(m, init_cell, dest_cell, time_s)
    
    # print("Path length : " + str(len(path)))
    # print(path)