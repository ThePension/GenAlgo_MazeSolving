from individual import Individual
import numpy as np

import time

if __name__ == "__main__":
    ind = Individual.randomIndividual(0, 0, 10, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    print(ind.adn)
    
    # time.sleep(5)
    
    mutated_ind = ind
    
    mutated_ind.mutate(0.5)
    
    print(mutated_ind.adn)