from copy import deepcopy
from random import randint
import numpy as np
import math

class Individual():
    def __init__(self, init_cell : tuple[int, int], adn = [], maze = [[]]):
        self.adn = adn
        self.init_cell = init_cell
        self.fitness = 10000
        self.maze = maze
        self.target_reached = False

    @staticmethod
    def isPathValid(path, maze):
        for cx, cy in path:
            # Check up and left boundaries
            if cx < 0 or cy < 0:
                return False
            
            # Check down and right boundaries
            if cx >= len(maze) or cy >= len(maze[0]):
                return False

            # Check it has hit a wall
            if maze[cx][cy] == 1:
                return False
            
        return True

    @staticmethod
    def randomIndividual(init_cell : tuple[int, int], adn_size : int, maze : np.ndarray):
        random_adn = [randint(0, 3) for i in range(0, adn_size)] 

        return Individual(init_cell, random_adn, maze)

    def clone(self):
        return Individual(self.init_cell, deepcopy(self.adn), self.maze)

    def mutate(self, gene_mutation_rate):
        mutated_adn = deepcopy(self.adn)
        
        for i in range(0, len(self.adn)):
            if randint(0, 100) < gene_mutation_rate * 100:
                old_gene = self.adn[i]
                
                if randint(0, 1) == 0:
                    mutated_adn[i] = (old_gene + 1) % 4
                else:
                    mutated_adn[i] = (old_gene - 1) % 4
                
        self.adn = mutated_adn
        

    # Return the path without consecutive duplicates
    def extract_partial_path(self, target : tuple[int, int]) -> list[tuple[int, int]]:
        path = self.compute_complete_valid_path()

        # Remove consecutive duplicates
        path = [path[i] for i in range(len(path)) if i == 0 or path[i] != path[i - 1]]
        
        # If the ind reached the target
        if target in path:
            return path[:path.index(target) + 1]

        return path
 

    def compute_fitness(self, target : tuple[int, int]):
        path = self.extract_partial_path(target) # Get the path without consecutive duplicates
          
        # If the ind reached the target
        if target in path:
            # Get the path to the target
            path_to_target = path[:path.index(target)]

            # If this path is valid
            if Individual.isPathValid(path_to_target, self.maze):
                # The fitness is the number of steps to reach the target
                self.fitness = len(path_to_target)
            else:
                # The fitness is the number of steps to reach the target + the number of invalid steps
                self.fitness = len(path_to_target) + len([cell for cell in path_to_target if Individual.isPathValid([cell], self.maze) == False])
        # If the ind didn't reach the target
        else:
            # Get the closest cell to the target
            closest_cell = min(path, key = lambda cell : math.sqrt((cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2))

            # Fitness is the number of steps to reach the closest cell to the target
            self.fitness = path.index(closest_cell)

            # Fitness is the number of steps to reach the closest cell to the target + the distance to the target
            self.fitness += math.sqrt((closest_cell[0] - target[0]) ** 2 + (closest_cell[1] - target[1]) ** 2) * 5

            # Fitness is the number of steps to reach the closest cell to the target + the distance to the target + the number of invalid steps
            self.fitness += len([cell for cell in path if Individual.isPathValid([cell], self.maze) == False])
            

    def compute_complete_path(self):
        prev_cell =  self.init_cell
        next_cell = (0, 0)
        path = [prev_cell]

        for move in self.adn:
            if move == 0:
                next_cell = (prev_cell[0] - 1, prev_cell[1])
            elif move == 1:
                next_cell = (prev_cell[0] + 1, prev_cell[1])
            elif move == 2:
                next_cell = (prev_cell[0], prev_cell[1] - 1)
            elif move == 3:
                next_cell = (prev_cell[0], prev_cell[1] + 1)

            path.append(next_cell)
            prev_cell = next_cell

        return path


    def compute_complete_valid_path(self) -> list[tuple[int, int]]:
        prev_cell = self.init_cell
        next_cell = (0, 0)
        path = [prev_cell]

        for move in self.adn:
            if move == 0:
                next_cell = (prev_cell[0] - 1, prev_cell[1])
            elif move == 1:
                next_cell = (prev_cell[0] + 1, prev_cell[1])
            elif move == 2:
                next_cell = (prev_cell[0], prev_cell[1] - 1)
            elif move == 3:
                next_cell = (prev_cell[0], prev_cell[1] + 1)

            if Individual.isPathValid([next_cell], self.maze):
                path.append(next_cell)
                prev_cell = next_cell
            else:
                path.append(prev_cell)

        return path