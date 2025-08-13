from genetic_algorithm.methods.selection.base_selection import SelectionMethod
from utilities.ga_utils import randf


class RandomWalkSelection(SelectionMethod):
    """ select individuals of population if random draw is smaller than
        some probability-rate (like walking a probabilistic tree structure) """

    def __init__(self, walk_rate):
        self.walk_rate = walk_rate

    def select(self, population, size):
        new_pop = []
        while len(new_pop) < size:
            if len(new_pop) > 0:
                if new_pop[-1] in population:
                    population.remove(new_pop[-1])
            for individual in population:
                if randf() < self.walk_rate:
                    new_pop.append(individual)
                    break
        return new_pop