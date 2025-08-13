from genetic_algorithm.methods.selection.base_selection import SelectionMethod
from utilities.ga_utils import randf


class RouletteSelection(SelectionMethod):
    """Performs roulette wheel selection, where selection probability is proportional to fitness."""

    def select(self, population, size):
        new_pop = []
        while len(new_pop) < size:
            if len(new_pop) > 0:  # Only change. If-block could equivalently be placed after for loop
                if new_pop[-1] in population:
                    population.remove(new_pop[-1])
            total_fitness = sum([individual.get_fitness() for individual in population])
            ball = randf() * total_fitness
            current_fitness = 0
            for individual in population:
                current_fitness += individual.get_fitness()
                if ball < current_fitness:
                    new_pop.append(individual)
                    break
        return new_pop