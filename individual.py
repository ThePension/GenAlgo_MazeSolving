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
    def nb_wall_hit(path, maze):
        nb_wall_hit = 0

        for cx, cy in path:
            if maze[cx][cy] == 1:
                nb_wall_hit += 1

        return nb_wall_hit

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
                
                # print("Mutation at index ", i, " from ", old_gene, " to ", mutated_adn[i])
                
        self.adn = mutated_adn
        

    # Return the path without consecutive duplicates
    def extract_partial_path(self, target : tuple[int, int]) -> list[tuple[int, int]]:
        path = self.compute_complete_valid_path()

        # Remove consecutive duplicates
        path = [path[i] for i in range(len(path)) if i == 0 or path[i] != path[i - 1]]
        
        # If the ind reached the target
        if target in path:
            return path[:path.index(target) + 1]
        # Remove consecutive pairs of duplicates
        
        # path = [path[i] for i in range(0, len(path) - 4) if path[i] != path[i + 2] or path[i + 1] != path[i + 3]]

        return path

        if target in path:
            # Get the index of the last initial cell in the path (must be before the first target in path)
            last_init_cell_index = path.index(init_cell, 0, path.index(target) + 1)
            print("last_init_cell_index ", last_init_cell_index)

            # Extract the path from the target to the initial cell
            extracted_path = path[last_init_cell_index : path.index(target) + 1]

            return extracted_path

        return path
        

    def compute_fitness(self, target : tuple[int, int]):
        path = self.extract_partial_path(target) # Get the path without consecutive duplicates
        
         # Get the closest cell to the target
        closest_cell = min(path, key = lambda cell : math.sqrt((cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2))
        self.fitness = math.sqrt((closest_cell[0] - target[0]) ** 2 + (closest_cell[1] - target[1]) ** 2) * 5 + path.index(closest_cell)
        
        return
    
        # If the ind reached the target
        if target in path:
            # Get the path to the target
            path_to_target = path[:path.index(target)]

            # If this path is valid
            if Individual.isPathValid(path_to_target, self.maze):
                # The fitness is the number of steps to reach the target
                self.fitness = len(path_to_target) - 1
            else:
                # The fitness is the number of steps to reach the target + the number of invalid steps
                self.fitness = len(path_to_target) - 1 + len([cell for cell in path_to_target if Individual.isPathValid([cell], self.maze) == False])
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
            
        # print("Fitness : ", self.fitness)

        # if target in path:
        #     extracted_path = self.extract_best_path(target)

        #     if Individual.isPathValid(extracted_path, self.maze):
        #         # Fitness is the number of steps to reach the target
        #         self.fitness = extracted_path.index(target)
        #         return

        # if not Individual.isPathValid(path, self.maze):
        #     self.fitness = 100000
        #     return

        # # Get the closest cell to the target
        # closest_cell_x, closest_cell_y = min(path, key=lambda x: math.sqrt((x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2))

        # target_x, target_y = target

        # nb_wall_hit = Individual.nb_wall_hit(path, self.maze)

        # self.fitness = (abs(closest_cell_x - target_x) + abs(closest_cell_y - target_y)) * 100 + nb_wall_hit * 100

        # if target in path:
        #     self.fitness = path.index(target)
        # else:
        #     closest_cell_x, closest_cell_y = min(path, key=lambda x: math.sqrt((x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2))
        #     manhattan_dist = (math.sqrt((closest_cell_x - target[0]) ** 2 + (closest_cell_y - target[1]) ** 2)) * 100
        #     self.fitness = manhattan_dist

        # final_cell = path[-1]

        # manhattan_dist = (math.sqrt((final_cell[0] - target[0]) ** 2 + (final_cell[1] - target[1]) ** 2)) * 100

        # self.fitness = manhattan_dist


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