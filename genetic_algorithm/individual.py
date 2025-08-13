from copy import copy

# ============= Genotype =============#


class Individual:
    next_id = 0

    def __init__(self, ga_config, genome=None, origin=None, parents=None):
        self.ga_config = ga_config
        self.genome = genome if genome is not None else ga_config.generate()
        self.fitness = None
        self.id = Individual.next_id
        self.age = 0
        self.origin = origin
        self.parents = parents
        Individual.next_id += 1

    def clone(self):  # ->Individual
        # The type(self) creates a new object of type Individual with arguments specified in the tuple afterwards
        new_individual = type(self)(self.ga_config, copy(self.genome), "clone", [self])
        return new_individual

    def mutate(self):  # ->Individual
        new_individual = type(self)(self.ga_config, self.ga_config.mutate(self.genome),
                                    "mutate", [self])
        return new_individual

    def recombine(self, other):  # ->Individual
        new_individual = type(self)(self.ga_config,
                                    self.ga_config.recombine(self.genome, other.genome),
                                    "recombine", [self, other])
        return new_individual

    def evaluate(self) -> float:
        self.fitness = self.compute_fitness()
        return self.fitness

    def get_fitness(self) -> float:
        return self.fitness

    def compute_fitness(self) -> float:
        return self.compute_problem_fitness()

    def compute_problem_fitness(self) -> float:
        return self.ga_config.evaluate_individual(self.genome)

    def to_str(self, fitness) -> str:
        result = ''
        result += '<' + str(self.__class__.__name__)
        result += ' ' + str(self.genome)
        result += ' ' + self.origin
        result += '\t@' + str(round(fitness, 5))
        result += '>'
        return result

    def to_str_fitness(self, population, quality_metric):
        self.evaluate(quality_metric)
        return self.to_str(self.fitness)

    def __str__(self) -> str:
        return self.to_str(self.fitness)
