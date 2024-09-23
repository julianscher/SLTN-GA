import os
import sys
from copy import copy
from pathlib import Path
from src.neural_network.NN import get_model
from src.neural_network.reinitialization_methods import reinitialize_parameters2
from src.neural_network.helper_functions import get_network_architecture
from typing import List
from src.genetic_algorithm.evaluate_methods import evaluate_population_method1, evaluate_population_method2, \
    evaluate_individual_method1, evaluate_population_method3
from src.genetic_algorithm.generation import generate_bit_vector, generate_lax_connectivity
from src.genetic_algorithm.helper_functions import *
from src.genetic_algorithm.mutation_methods import single_point_mutation
from src.genetic_algorithm.recombination_methods import crossover_recombination
from src.genetic_algorithm.selection_methods import *

sys.path.append(str(Path(__file__).parent.parent)+"/src/genetic_algorithm")
rel_path = "resources/neural_network"
path = Path(os.path.join(Path(__file__).parent.parent.parent, rel_path))


class Individual:
    next_id = 0

    def __init__(self, problem_instance, genome=None, origin=None, parents=None, **kwargs):
        self.args = kwargs
        # Store the arguments for the population and the problem instance together in a new dictionary
        environment_args = dict(self.args, **problem_instance.args)
        # Rule: pass method arguments in the deepest call
        self.mutate_args = get_method_args(environment_args, self.args.get("mutate_args", {}), "mutate_args")
        self.recombine_args = get_method_args(environment_args, self.args.get("recombine_args", {}), "recombine_args")
        self.generate_args = get_method_args(environment_args, self.args.get("generate_args", {}), "generate_args")
        self.evaluate_individual_args = get_method_args(environment_args, self.args.get("evaluate_individual_args", {}),
                                                        "evaluate_individual_args")
        self.problem_instance = problem_instance
        self.genome = genome or problem_instance.generate(self.generate_args)
        self.fitness = None
        self.id = Individual.next_id
        self.age = 0
        self.origin = origin
        self.parents = parents
        Individual.next_id += 1

    def clone(self):  # ->Individual
        # The type(self) creates a new object of type Individual with arguments specified in the tuple afterwards
        new_individual = type(self)(self.problem_instance, copy(self.genome), "clone", [self], **self.args)
        return new_individual

    def mutate(self):  # ->Individual
        new_individual = type(self)(self.problem_instance, self.problem_instance.mutate(self.genome, self.mutate_args),
                                    "mutate", [self], **self.args)
        return new_individual

    def recombine(self, other):  # ->Individual
        new_individual = type(self)(self.problem_instance,
                                    self.problem_instance.recombine(self.genome, other.genome, self.recombine_args),
                                    "recombine", [self, other], **self.args)
        return new_individual

    def evaluate(self, population) -> float:
        self.fitness = self.compute_fitness(population)
        return self.fitness

    def get_fitness(self) -> float:
        return self.fitness

    def compute_fitness(self, population) -> float:
        return self.compute_problem_fitness(population)

    def compute_problem_fitness(self, population) -> float:
        return self.problem_instance.evaluate_individual(self.genome, population, self.evaluate_individual_args)

    def to_str(self, fitness) -> str:
        result = ''
        result += '<' + str(self.__class__.__name__)
        result += ' ' + str(self.genome)
        result += ' ' + self.origin
        result += '\t@' + str(round(fitness, 5))
        result += '>'
        return result

    def to_str_accuracy_fitness(self, population):
        self.evaluate(population)
        return self.to_str(self.fitness)

    def __str__(self) -> str:
        return self.to_str(self.fitness)


class Population:
    def __init__(self, problem_instance, individuals=None, **kwargs):
        self.args = kwargs
        self.problem_instance = problem_instance

        # individuals can either be a list of previous generation individuals (which we adopt for this population)
        # or just a individual-type class, in which case we generate a new population of these class instances 
        if type(individuals) == list:
            self.individual_class = kwargs.get('individual_class',
                                               (len(individuals) > 0) and type(individuals[0]) or Individual)
            self.individuals = individuals
        elif type(individuals) == int:
            self.individual_class = kwargs.get('individual_class', Individual)
            self.individuals = self.generate_all(individuals)
        else:
            self.individuals = []
        self.size = len(self.individuals)
        self.age = 0
        self.mig_rate = self.args.get('mig', 0.1)  # "migration rate"
        self.mut_rate = self.args.get('mut', 0.1)  # "mutation rate": The likelihood for a mutation
        self.rec_rate = self.args.get('rec', 0.3)  # "recombination rate": How many individuals should recombine
        self.par_rate = self.args.get('par', 0.3)  # "parents rate": How many individuals are there to mate with
        self.selection = self.args.get('selection', select_by_cutoff)  # default method is select_by_cutoff
        self.selection_args = self.args.get('selection_args', {})
        # Store the arguments for the population and the problem instance together in a new dictionary
        environment_args = dict(self.args, **problem_instance.args)
        self.evaluate_for_selection_args = get_method_args(environment_args,
                                                           self.args.get("evaluate_for_selection_args", {}),
                                                           "evaluate_for_selection_args")
        self.evaluate_for_recombination_args = get_method_args(environment_args,
                                                               self.args.get("evaluate_for_recombination_args", {}),
                                                               "evaluate_for_recombination_args")

    def choose_mate(self, individual) -> Individual:
        """ Parent selection """
        return self.individuals[randi(max=int(self.par_rate * self.size))]

    def choose_random(self) -> Individual:
        return self.individuals[randi(max=self.size)]

    def generate(self, origin="generate"):
        return self.individual_class(self.problem_instance, origin=origin, **self.args)

    def generate_all(self, size):
        return [self.generate() for _ in range(size)]

    def evaluate_for_selection(self):
        return self.problem_instance.evaluate_for_selection(self, self.evaluate_for_selection_args)

    def evaluate_for_recombination(self):
        return self.problem_instance.evaluate_for_recombination(self, self.evaluate_for_recombination_args)

    def evaluate_by_accuracy(self):
        return evaluate_population_method1(self, {"environment_args": dict(self.args, **self.problem_instance.args)})

    def evaluate_accuracy_on_test_dataset(self, X_test, Y_test):
        environment_args = dict(self.args, **self.problem_instance.args)
        environment_args["X_for_accuracy"] = X_test
        environment_args["Y_for_accuracy"] = Y_test
        # Set individuals in population to test mode
        for individual in self.individuals:
            individual.evaluate_individual_args['environment_args']['X_for_accuracy'] = X_test
            individual.evaluate_individual_args['environment_args']['Y_for_accuracy'] = Y_test
            individual.evaluate_individual_args['environment_args']['use_stored'] = False
        return evaluate_population_method3(self, {"environment_args": environment_args})

    def select(self, size=None):
        """ Survivor selection """
        size = size or self.size  # select as many individuals as is specified in size
        self.evaluate_for_selection()
        self.individuals = self.selection(self.individuals, size, **self.selection_args)
        return self

    def mutate(self):
        self.individuals += [individual.mutate() for individual in self.individuals if randb(self.mut_rate)]
        return self

    def recombine(self):
        self.evaluate_for_recombination()
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


class OptimizationProblem:
    """ general problem interface """

    def maximizing(self):
        return self.objective_maximizing

    def minimizing(self):
        return not self.objective_maximizing

    def best(self):
        return None

    def best_result(self):
        return None

    def dim(self):
        return self.dimensionality

    def dims(self):
        return range(0, self.dim())


class WinningTicketProblem(OptimizationProblem):
    def __init__(self, seed=None, model=get_model([2, 20, 1], None, reinitialize_parameters2),
                 generate_method=generate_lax_connectivity, mutate_method=single_point_mutation,
                 recombine_method=crossover_recombination,
                 evaluate_population_method_for_selection=evaluate_population_method1,
                 evaluate_population_method_for_recombination=evaluate_population_method2,
                 evaluate_individual_method=evaluate_individual_method1, **kwargs):

        self.args = kwargs
        self.objective_maximizing = True
        self.NN_architecture = get_network_architecture(model)
        self.model = model
        self.args["seed"] = seed
        self.args["NN_architecture"] = self.NN_architecture
        self.args["model"] = model
        self.dimensionality = calculate_dimensionality(self.NN_architecture)
        self.generate_method = generate_method
        self.mutate_method = mutate_method
        self.recombine_method = recombine_method
        self.evaluate_population_method_for_selection = evaluate_population_method_for_selection
        self.evaluate_population_method_for_recombination = evaluate_population_method_for_recombination
        self.evaluate_individual_method = evaluate_individual_method

        if len(self.NN_architecture) <= 1:
            raise ValueError("NN_architecture not supported")

    def generate(self, generate_args) -> List[float]:
        # generate one valid genome that conforms to the problem requirements
        return self.generate_method(generate_args)

    def min(self) -> List[int]:
        return generate_bit_vector({"environment_args": self.args, "generate_args": {"arg": "min"}})

    def max(self) -> List[int]:
        return generate_bit_vector({"environment_args": self.args, "generate_args": {"arg": "max"}})  # the original dense network

    def mutate(self, genome, mutate_args):
        return self.mutate_method(genome, mutate_args)

    def recombine(self, genome1, genome2, recombine_args):
        return self.recombine_method(genome1, genome2, recombine_args)

    def best(self) -> List[int]:
        pass

    def best_result(self) -> float:
        pass

    def evaluate_for_selection(self, population, evaluate_for_selection_args):
        return self.evaluate_population_method_for_selection(population, evaluate_for_selection_args)

    def evaluate_for_recombination(self, population, evaluate_for_recombination_args):
        return self.evaluate_population_method_for_recombination(population, evaluate_for_recombination_args)

    def evaluate_individual(self, genome, population, evaluate_individual_args):
        return self.evaluate_individual_method(genome, population, evaluate_individual_args)


