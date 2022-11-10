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

    population_size = 50
    # Longest path possible is the number of cells in the grid / 2
    h, w = grid.shape
    adn_size = round(w * h * 2)
    mutation_rate = 0.5 # Probability of mutation for a individual
    gene_mutation_rate = 0.1 # Probability of mutation for a gene
    mating_rate = 0.5 # Probability of crossover for a pair of individuals
    ellitiste_mutation_rate = 0.1
    gen_count = 0

    population = [Individual.randomIndividual(start_cell[0], start_cell[1], adn_size, grid) for i in range(0, population_size)]
    best_ind = population[0] # Keep track of the best path found for the moment

    while(True):
        # ------------------------------------------
        #
        # Compute the next generation
        #
        # ------------------------------------------
        ind, gen_count = compute_generation(population, grid, end_cell, mutation_rate, mating_rate, gene_mutation_rate, gen_count)

        ind.compute_fitness(end_cell)

        if (ind.fitness < best_ind.fitness):
            best_ind = ind

        print("Generation: " + str(gen_count) + " - Best path fitness: " + str(best_ind.fitness))

        # ------------------------------------------
        #
        # Check if the algorithm has to stop
        #
        # ------------------------------------------
        if time.time() - start_time >= max_time_s:
            break

    return best_ind.extract_partial_path(end_cell)

def compute_generation(population:list[Individual], grid:np.ndarray, target:tuple, mutation_rate : float, mating_rate : float, gene_mutation_rate : float, ellitiste_mutation_rate : float, gen_count = 0) -> Individual:
    gen_count += 1
    offspring = selection(population, int(len(population) / 2))

    bests = offspring[round(ellitiste_mutation_rate * len(population))]

    for ind in offspring:
        ind.clone()

    for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
        if randint(0, 100) < mating_rate * 100:
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    for ind in offspring:
        if randint(0, 100) < mating_rate * 100:
            ind.mutate(gene_mutation_rate)
            
        ind.compute_fitness(target)

    population[:] = offspring

    # shuffle the population to improve diversity
    np.random.shuffle(population)

    # Get the 5 best individuals
    bestPaths = selection(population, 5)

    [print("Best path fitness in this gen : " + str(ind.fitness)) for ind in bestPaths]

    return bestPaths[0].clone(), gen_count # return the best individual
    

def selection(population:list[Individual], k : int) -> list[Individual]:
    return sorted(population, key=lambda x: x.fitness, reverse=False)[:k]

def crossover(parent1 : Individual, parent2 : Individual) -> tuple[Individual, Individual]:
    adn_parent1 = parent1.clone().adn
    adn_parent2 = parent2.clone().adn

    crossing_point = (len(adn_parent1) / 2)

    adn_child1 = adn_parent1[crossing_point:] + adn_parent2[:crossing_point]
    adn_child2 = adn_parent1[crossing_point:] + adn_parent1[:crossing_point]

    for i in range(0, len(adn_parent1)):
        if randint(0, 1) == 0:
            adn_child1.append(adn_parent1[i])
            adn_child2.append(adn_parent2[i])
        else:
            adn_child1.append(adn_parent2[i])
            adn_child2.append(adn_parent1[i])

    child1 = Individual(parent1.x, parent1.y, adn_child1)
    child2 = Individual(parent2.x, parent2.y, adn_child2)

    return child1, child2   

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

    # display_labyrinth(grid, start, end, path)