"""Main class for solving labyrinths with genetic algorithms.

Tested Python 3.9+
"""

# Sauces :
# - https://www.diva-portal.org/smash/get/diva2:927325/FULLTEXT01.pdf

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
from individual import Individual
from mazegenerator import MazeGenerator
from deap import base, creator, tools, algorithms
import math

# ------------------------------------------
#
# Global variables
#
# ------------------------------------------
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

# ------------------------------------------
#
# Initialize population
#
# ------------------------------------------

def random_individual(adn_size : int):
    random_adn = [randint(0, 3) for i in range(0, adn_size)]
    return creator.Individual(random_adn)

def initialize_population(adn_size : int, n : int):
    return [random_individual(adn_size) for i in range(0, n)]


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
    adn_size = int((w * h) / 2)
    population_size =  w + h # int((w + h) * (4 / 3))
    mutation_rate = 0.8 # Probability of mutation for a individual
    gene_mutation_rate = 0.1 # Probability of mutation for a gene
    mating_rate = 0.5 # Probability of crossover for a pair of individuals
    ellitiste_mutation_rate = 0.01
    gen_count = 0


    # ------------------------------------------
    #
    # Toolbox registering
    #
    # ------------------------------------------
    
    toolbox.register("crossover", tools.cxOnePoint)
    toolbox.register("mutate", mutate, gene_mutation_rate=gene_mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", compute_fitness, init_cell = start_cell, maze = grid, target = end_cell)
    toolbox.register("population", initialize_population)

    # ------------------------------------------
    #
    # Initialization
    #
    # ------------------------------------------
    
    population = toolbox.population(n=population_size, adn_size=adn_size)
    best_ind = population[0] # Keep track of the best path found for the moment

    best_ind.fitness.values = toolbox.evaluate(best_ind, init_cell=start_cell, maze=grid, target=end_cell)
    
    while(True):
        # -----------------------------------------
        #
        # Compute the next generation
        #
        # ------------------------------------------

        ind, new_population = compute_generation(population, grid, start_cell, end_cell, mutation_rate, mating_rate, gene_mutation_rate, ellitiste_mutation_rate)
        
        gen_count += 1
        
        population[:] = new_population

        # fit = compute_fitness(ind, init_cell=start_cell, maze=grid, target=end_cell)
        # ind.fitness = fit

        if (ind.fitness.values[0] < best_ind.fitness.values[0]):
            best_ind = toolbox.clone(ind)

        # print("NPOP : " + str(len(population)) + " | Generation: " + str(gen_count) + " - Best path fitness: " + str(best_ind.fitness.values[0]))

        # ------------------------------------------
        #
        # Check if the algorithm has to stop
        #
        # ------------------------------------------
        if time.time() - start_time >= max_time_s:
            break

    return compute_subpath(best_ind, start_cell, grid, end_cell)

def compute_generation(population:list[Individual], grid:np.ndarray, init_cell : tuple[int, int], target:tuple, mutation_rate : float, mating_rate : float, gene_mutation_rate : float, ellitiste_mutation_rate : float) -> Individual:
    ellitiste_number = round(ellitiste_mutation_rate * len(population))

    bests = population[:ellitiste_number:]
    population = population[ellitiste_number::]
    
    offspring = toolbox.select(population, len(population))

    offspring = [toolbox.clone(ind) for ind in offspring]

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if randint(0, 100) < mating_rate * 100:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if randint(0, 100) < mutation_rate * 100:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    # fitnesses = [compute_fitness(ind, init_cell=init_cell, maze=grid, target=target) for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population = bests + offspring

    # shuffle the population to improve diversity
    np.random.shuffle(population)

    # Get the 5 best individuals
    bestPaths = tools.selBest(population, k=5)

    # [print("Best path fitness in this gen : " + str(ind.fitness)) for ind in bestPaths]

    return bestPaths[0], population # return the best individual


# ------------------------------------------
#
# Mutation function
#
# ------------------------------------------

def mutate(ind : creator.Individual, gene_mutation_rate : float = 0.01) -> creator.Individual:   
    for i in range(0, len(ind)):
        if randint(0, 100) < gene_mutation_rate * 100:
            old_gene = ind[i]
            
            if randint(0, 1) == 0:
                ind[i] = (old_gene + 1) % 4
            else:
                ind[i] = (old_gene - 1) % 4
            
    return (creator.Individual(ind),)


# ------------------------------------------
#
# Fitness computation function
#
# ------------------------------------------

def compute_fitness(ind : creator.Individual, init_cell : tuple[int, int], maze : np.ndarray, target : tuple[int, int]):
    path = compute_subpath(ind, init_cell, maze, target) # Get the path without consecutive duplicates

    #     # Fitness / calcul du score
    # if target in path:
    #     return (path.index(target),)
    # else:
    #     return (len(ind) + (abs(target[0] - path[-1][0]) + abs(target[1] - path[-1][1])),)

    fitness = None

    # If the ind reached the target
    if target in path:
        # Get the path to the target
        path_to_target = path[:path.index(target)]

        # The fitness is the number of steps to reach the target
        fitness = len(path_to_target)
        
    # If the ind didn't reach the target
    else:
        # Fitness is the number of steps to reach the closest cell to the target (which is the last one in the path)
        fitness = len(path)

        # + the distance to the target
        fitness += math.sqrt((path[-1][0] - target[0]) ** 2 + (path[-1][1] - target[1]) ** 2) * 10

    return (fitness,) # Must be a tuple
 

# ------------------------------------------
#
# 
#
# ------------------------------------------

def isPathValid(path, maze):
    prev_cx, prev_cy = None, None

    # print("Path : " + str(path))

    for cx, cy in path:
        # Check up and left boundaries
        if cx < 0 or cy < 0:
            return False
        
        # Check down and right boundaries
        if cx >= len(maze) or cy >= len(maze[0]):
            return False

        # Check if it has hit a wall
        if maze[cx][cy] == 1:
            return False

        # Check if it's returning back (loop)
        if cx == prev_cx and cy == prev_cy:
            return False

        prev_cx, prev_cy = cx, cy
        
    return True

# ------------------------------------------
#
# Compute chromosome function - Return the path based on the genes in the chromosome
# - Remove consecutive duplicates
# - If the ind reached the target before the end of the chromosome, return this subpath
# - If the ind did not reach the target before the end of the chromosome,
#   return the path leading to the closest (euclid distance) cell of the target
#
# ------------------------------------------

def compute_subpath(ind : creator.Individual, init_cell : tuple[int, int], maze : np.ndarray, target : tuple[int, int]) -> list[tuple[int, int]]:
    path = compute_complete_valid_path(ind, init_cell, maze, target)

    # Remove consecutive duplicates
    path = [path[i] for i in range(len(path)) if i == 0 or path[i] != path[i - 1]]
    
    # If the ind reached the target
    if target in path:
        return path[:path.index(target) + 1]
    
    closest_cell = min(path, key = lambda cell : math.sqrt((cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2))

    return path[:path.index(closest_cell) + 1]

# ------------------------------------------
#
#   Get the move, as a tuple[int, int],
#   based on the gene value, described below
#
#   0 : Move left   (-1, +0)
#   1 : Move right  (+1, +0)
#   2 : Move down   (+0, -1)
#   3 : Move up     (+0, +1)
#
# ------------------------------------------

def getMoveFromGene(gene : int) -> tuple[int, int]:
    if gene == 0:
        return (-1, 0)
    elif gene == 1:
        return (1, 0)
    elif gene == 2:
        return (0, -1)
    elif gene == 3:
        return (0, 1)
    else:
        raise Exception("Invalid gene")

# ------------------------------------------
#
# Compute chromosome function - Return the path based on the genes in the chromosome
# - Does not apply gene that leads to an invalid path (wall, out of bounds)
# - The path may contain consecutive duplicates, or loops
# - Tries the prevent the ind from going back to the previous cell
# - Tries to prevent the ind from reaching dead ends (cul de sac)
#
# ------------------------------------------

def compute_complete_valid_path(ind : creator.Individual, init_cell : tuple[int, int], maze : np.ndarray, target : tuple[int, int]) -> list[tuple[int, int]]:
    maze = maze.copy()
    prev_cell = init_cell
    current_cell = init_cell
    next_cell = (0, 0)
    path = [init_cell]

    for gene in ind:
        is_current_cell_in_deadend = True

        for i in range(0, 4):
            move_x, move_y = getMoveFromGene((gene + i) % 4)
            # print("\nmove_x : " + str(move_x) + " move_y : " + str(move_y))

            next_cell = (current_cell[0] + move_x, current_cell[1] + move_y)
            # print("Leads from (" + str(current_cell[0]) + ", " + str(current_cell[1]) + ") to (" +str(next_cell[0]) + ", " + str(next_cell[1]) + ")")
            
            if isPathValid([prev_cell, next_cell], maze):
                path.append(next_cell)

                # print("Valid move")

                if next_cell == target:
                    return path

                prev_cell = current_cell
                current_cell = next_cell
                is_current_cell_in_deadend = False
                break # Break the nested loop
            # else:
                # print("Not valid")

        # If there is no move available, the current cell is in a dead end (cul de sac)
        if is_current_cell_in_deadend:
            # print("Dead end")
            # Prevent returning to this dead end by adding a wall
            maze[current_cell[0]][current_cell[1]] = 1
            path.append(prev_cell)
            current_cell = prev_cell
            prev_cell = current_cell

    return path


if __name__ == "__main__":
    import numpy as np

    file = "grid10.npy"
    grid = np.load(file)
    start = (0, 0)
    h = grid.shape[0]
    w = grid.shape[1]
    end = (h - 1, w - 1)
        
    best_path = solve_labyrinth(grid, start, end, 1)

    print("Solution found in " + str(len(best_path) - 1) + " steps")
    print(best_path)

    
    # size = 15
    # init_cell = (1, 1)
    # dest_cell = (size - 2, size - 2)
    # time_s = 15
    
    # m = MazeGenerator.generate_maze(15, 15)

    # path = solve_labyrinth(m, init_cell, dest_cell, time_s)
    
    # print("Path length : " + str(len(path)))
    # print(path)