import random
import time

import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


def display_labyrinth(grid, start_cell, end_cell, solution=None):
    """Display the labyrinth matrix and possibly the solution with matplotlib.
    Free cell will be in light gray.
    Wall cells will be in dark gray.
    Start and end cells will be in dark blue.
    Path cells (start, end excluded) will be in light blue.
    :param grid np.array: labyrinth matrix
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
    plt.show()


# List des directions possible
NORTH = (-1, 0)
EAST = (0, 1)
WEST = (0, -1)
SOUTH = (1, 0)

DIRECTION = [NORTH, EAST, WEST, SOUTH]


def solve_labyrinth(grid, start_cell, end_cell, max_time_s):
    """Attempt to solve the labyrinth by returning the best path found
    :param grid np.array: numpy 2d array
    :start_cell tuple: tuple of i, j indices for the start cell
    :end_cell tuple: tuple of i, j indices for the end cell
    :max_time_s float: maximum time for running the algorithm
    :return list: list of successive tuple i, j indices who forms the path
    """

    # Initialisation des constantes
    CHROMOSOME_LENGTH = (grid.shape[0] + grid.shape[1]) * 3  # Taille d'un chromosone
    POPULATION_SIZE = grid.shape[0] + grid.shape[1]  # Taille de la population
    CXPB = 0.7  # Probabilité de crossover
    MUTPB = 0.1  # Probabilité de crossover
    INDPB = 0.1  # Independent probability for each attribute to be exchanged to another position. (https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py#L105)

    tb = base.Toolbox()  # Init toolbox

    # Init Fitness et individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Register de mate, mutate, select et evaluate
    tb.register("mate", tools.cxOnePoint)
    tb.register("mutate", tools.mutShuffleIndexes, indpb=INDPB)
    tb.register("select", tools.selBest)
    tb.register("evaluate", evaluate_individual)

    # Register d'initialisation de gene, individu et population
    tb.register("init_gene", random.randint, 0, 3)
    tb.register("init_individual", tools.initRepeat, creator.Individual, tb.init_gene, CHROMOSOME_LENGTH)
    tb.register("init_population", tools.initRepeat, list, tb.init_individual)

    # Creation de la population et du hallOfFame
    population = tb.init_population(POPULATION_SIZE)
    hallOfFame = tools.HallOfFame(1)

    # Premiere evaluation de la population
    for ind in population:
        tb.evaluate(grid, start_cell, end_cell, ind)
    hallOfFame.update(population)

    # Debut de "l'entrainement" de l'algorithme
    # Inspirer de eaSimple (https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L85)
    remainingTime = max_time_s
    while remainingTime > 0:
        t1 = time.perf_counter()

        # Select
        offspring = tb.select(population, len(population))

        # Crossover et mutation (https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L33)
        offspring = algorithms.varAnd(offspring, tb, CXPB, MUTPB)

        # Evaluation de la nouvelle population
        for ind in offspring:
            tb.evaluate(grid, start_cell, end_cell, ind)

        # Ajout du meilleur dans le hallOfFame (si meilleur que celui déjà présent)
        hallOfFame.update(offspring)

        # Remplacement de l'ancienne population par la nouvelle
        population = offspring

        t2 = time.perf_counter() - t1
        remainingTime -= t2

        # print("Best fitness: {}".format(hallOfFame[0].fitness.values[0]))

    # Retourne le path du meilleur jamais trouver
    return computePath(grid, start_cell, end_cell, hallOfFame[0])

# Fonction d'evalutation d'un individu
def evaluate_individual(grid, startCell, endCell, individual):
    """
    :param grid np.array: numpy 2d array
    :start_cell tuple: tuple of i, j indices for the start cell
    :end_cell tuple: tuple of i, j indices for the end cell
    :individual Individual: a list of direction and a fitness
    """
    # Creation du chemin valide avec les directions aléatoires de l'individu
    tempGrid = grid.copy()
    path = computePath(tempGrid, startCell, endCell, individual)

    # Fitness / calcul du score
    if endCell in path:
        individual.fitness.values = (path.index(endCell),)
    else:
        individual.fitness.values = len(individual) + (
                abs(endCell[0] - path[-1][0]) + abs(endCell[1] - path[-1][1])),

# Fonction qui creer un path valide avec les directions d'un individu
def computePath(grid, startCell, endCell, individual):
    """
    :param grid np.array: numpy 2d array
    :start_cell tuple: tuple of i, j indices for the start cell
    :end_cell tuple: tuple of i, j indices for the end cell
    :individual Individual: a list of direction and a fitness
    :return list: list of successive tuple i, j indices who forms the path
    """
    #Initialisation
    path = [startCell]
    actualCell = startCell
    previousCell = startCell

    for direction in individual:
        # Validation et ajout de la direction
        cell, direction = validDirection(grid, actualCell, direction, previousCell)
        path.append(cell)

        # Arret si destination trouvee
        if cell == endCell:
            break

        previousCell = actualCell
        actualCell = cell
    return path

# Fonction qui valide si le mouvement est possible, si impossible cherche un mouvement possible
def validDirection(grid, actualCell, direction, previousCell):
    """
    :param grid np.array: numpy 2d array
    :actualCell tuple: tuple of i, j indices for the actual cell of the path
    :direction int: index of the direction in list DIRECTION
    :previousCell tuple: tuple of i, j indices for the previous cell of the path
    :return tuple: tuple of the next valid cell to go
    """
    i = direction

    # Utilise les indexes inversé en python pour itérer sur toute la liste
    while i > (direction - len(DIRECTION)):
        nextCell = (actualCell[0] + DIRECTION[i][0], actualCell[1] + DIRECTION[i][1])
        gridShape = grid.shape
        nextX = nextCell[0]
        nextY = nextCell[1]

        # Test si valide
        if 0 <= nextX < gridShape[0] \
                and 0 <= nextY < gridShape[1] \
                and grid[nextX][nextY] == 0 \
                and nextCell != previousCell:
            return nextCell, i

        i -= 1

    # Creation mur fictif si aucune direction possible
    grid[actualCell[0]][actualCell[1]] = 1
    return previousCell, direction

def random_maze(w : int, h : int, wall_ratio : float) -> np.ndarray:
    maze = np.zeros((h, w), dtype=int)
    for i in range(0, h):
        for j in range(0, w):
            if np.random.rand() < wall_ratio:
                maze[i, j] = 1
    return maze

# Fonction main de test
if __name__ == '__main__':
    file = "grid10.npy"
    grid = np.load(file)
    # grid = random_maze(20, 20, 0.3)
    start = (0, 0)
    h = grid.shape[0]
    w = grid.shape[1]
    end = (h - 1, w - 1)

    solution = solve_labyrinth(grid, start, end, 1)
    print("Solution found in " + str(len(solution) - 1) + " steps")
    print(solution)
