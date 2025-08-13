from genetic_algorithm.methods.selection.base_selection import SelectionMethod


class GracefulCutoffSelection(SelectionMethod):
    """ select all individuals of population until cut-off, but retain (= behalten)
            all individuals younger than the grace-period age """

    def __init__(self, age_grace_period):
        self.age_grace_period = age_grace_period

    def select(self, population, size):
        new_pop = []
        for individual in population:
            if 1 <= individual.age <= self.age_grace_period:
                new_pop.append(individual)
        for new_individual in new_pop:
            population.remove(new_individual)  # remove all individuals that are part of new_pop from old pop
        new_pop += population
        # Now the individuals with age < age_grace_period are at the front and are therefore more likely to be included in
        # end_result -> Why not use sort() ?
        return new_pop[0:size]