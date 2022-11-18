# Course           : None
# Date             : 11.11.2022
# Author           : Aubert Nicolas
# Exercice         : None
# File description : Generate real maze using recursive backtracking algorithm

import random
import numpy as np

class MazeGenerator():

    # ------------------------------------------
    #
    # Return a np.array of size (width, height),
    # representing a maze, where
    #           1 = wall
    #           0 = empty cell (path)
    #
    # ------------------------------------------
    
    @staticmethod
    def generate_maze(width, height):
        maze = []
        visited = []
        
        # Start with a grid of walls
        for i in range(height):
            maze.append([])
            visited.append([])
            for j in range(width):
                maze[i].append(1)
                visited[i].append(False)

        # Pick a cell
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        return np.array(MazeGenerator.dfs((x, y), visited, maze))


    @staticmethod
    def dfs(cell, visited, maze):
        cx, cy = cell

        visited[cx][cy] = True
        maze[cx][cy] = 0

        neighbors = MazeGenerator.computeFrontierCells(cx, cy, len(maze), len(maze[0]))

        # Shuffle the neighbors
        random.shuffle(neighbors)

        for nx, ny in neighbors:
             if not visited[nx][ny]:
                # Connect the current cell with the neighbor by setting the cell in-between to state 0
                in_between_cell_offset_x = int(map_(nx - cx, -2, 2, -1, 1)) # {0, 1, -1}
                in_between_cell_offset_y = int(map_(ny - cy, -2, 2, -1, 1)) # {0, 1, -1}

                in_between_cell_x = cx + in_between_cell_offset_x
                in_between_cell_y = cy + in_between_cell_offset_y

                maze[in_between_cell_x][in_between_cell_y] = 0

                visited[in_between_cell_x][in_between_cell_y] = True

                MazeGenerator.dfs((nx, ny), visited, maze)

        return maze

    # ------------------------------------------
    #
    # Return a list of cells that are distant by 2 cells from the current cell
    #
    # ------------------------------------------
    
    @staticmethod
    def computeFrontierCells(x, y, w, h):
        neighbors = []
        offset = 2

        if (x + offset < w): neighbors.append([x + offset, y])
        if (x - offset >= 0): neighbors.append([x - offset, y])
        if (y + offset < h): neighbors.append([x, y + offset])
        if (y - offset >= 0): neighbors.append([x, y - offset])

        return neighbors

# Stolen to p5js
def map_(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2


# ------------------------------------------
#
# Example usage
#
# ------------------------------------------


if __name__ == "__main__":
    maze = MazeGenerator.generate_maze(20, 20)
    for row in maze:
        print(row)