from genetic_algorithm.individual import Individual
from utilities.ga_utils import randi, randb

class Population:
    def __init__(self, pop_size, mig_rate, mut_rate, rec_rate, par_rate, ga_config, individuals=None):
        self.ga_config = ga_config
        self.individual_class = Individual
        self.individuals = self.generate_all(pop_size) if individuals is None else individuals
        self.size = pop_size
        self.age = 0
        self.mig_rate = mig_rate  # "migration rate"
        self.mut_rate = mut_rate  # "mutation rate": The likelihood for a mutation
        self.rec_rate = rec_rate  # "recombination rate": How many individuals should recombine
        self.par_rate = par_rate  # "parents rate": How many individuals are there to mate with

    def choose_mate(self, individual) -> Individual:
        """ Parent selection """
        return self.individuals[randi(max=int(self.par_rate * self.size))]

    def choose_random(self) -> Individual:
        return self.individuals[randi(max=self.size)]

    def generate(self, origin="generate"):
        return self.individual_class(self.ga_config, origin=origin)

    def generate_all(self, size):
        return [self.generate() for _ in range(size)]

    def evaluate_survivors(self):
        return self.ga_config.evaluate_survivors(self)

    def evaluate_parents(self):
        return self.ga_config.evaluate_parents(self)

    def select(self, size=None):
        """ Survivor selection """
        size = size or self.size  # select as many individuals as is specified in size
        self.evaluate_survivors()
        self.individuals = self.ga_config.select(self.individuals, size)
        return self

    def mutate(self):
        self.individuals += [individual.mutate() for individual in self.individuals if randb(self.mut_rate)]
        return self

    def recombine(self):
        self.evaluate_parents()
        self.individuals += [individual.recombine(self.choose_mate(individual)) for individual in self.individuals if
                             randb(self.rec_rate)]
        return self

    def migrate(self):
        self.individuals += [self.generate("migrate") for _ in range(int(self.mig_rate * self.size))]
        return self

    def evolve(self):  # -> Population:

        # manage age counters (for possible usage in selection)
        self.age += 1
        for individual in self.individuals:
            individual.age += 1

        # do the evolution
        self.recombine()  # add new children to the population
        self.mutate()  # add new mutants to the population
        self.migrate()  # add new hypermutants to the population
        self.select()  # cut down population to its original size

        return self

    def __str__(self):
        # Only call after calling evolve() for current fitness
        return f"Population(t{self.age}):\n" + "\n".join(str(i) for i in self.individuals) + "\n"
