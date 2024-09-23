import dill
import numpy as np
import sys
from pathlib import Path
import os
from src.experiments.base.evol_routines import evol_routines_dict
from src.experiments.base.final_routines import final_routines_dict
from src.experiments.base.init_routines import init_routines_dict
from src.experiments.base.termination_conditions import termination_conditions_dict
from src.neural_network.helper_functions import get_network_architecture
from src.genetic_algorithm.GA import WinningTicketProblem, Population

sys.path.append(str(Path(__file__).parent))
rel_path = "resources/test_results"
path = Path(os.path.join(Path(__file__).parent.parent.parent.parent, rel_path))


class Experiment:
    """ Generic experiment framework """

    def __init__(self, name, seed, generate_method, mutate_method, mutate_connectivity_scheme, recombine_method,
                 recombine_connectivity_scheme, evaluate_population_method_for_selection,
                 evaluate_population_method_for_recombination, evaluate_individual_method, param_reinitialization_method,
                 model, X_for_accuracy, Y_for_accuracy, log):
        self.experiment_name = name
        self.log_path = f"{path}/{self.experiment_name}"
        self.log = log

        # Define all settings
        self.seed = seed
        self.generate_method = generate_method
        self.mutate_method = mutate_method
        self.mutate_connectivity_scheme = mutate_connectivity_scheme
        self.recombine_method = recombine_method
        self.recombine_connectivity_scheme = recombine_connectivity_scheme
        self.evaluate_population_method_for_selection = evaluate_population_method_for_selection
        self.evaluate_population_method_for_recombination = evaluate_population_method_for_recombination
        self.evaluate_individual_method = evaluate_individual_method
        self.param_reinitialization_method = param_reinitialization_method
        self.model = model
        self.X_for_accuracy = X_for_accuracy
        self.Y_for_accuracy = Y_for_accuracy

        # Log wtp settings
        if self.log:
            os.mkdir(self.log_path)
            with open(self.log_path + "/wtp_settings.txt", "w") as text_file:
                print("seed: " + str(self.seed), file=text_file)
                print("generate_method: " + self.generate_method.__name__, file=text_file)
                print("mutate_method: " + self.mutate_method.__name__, file=text_file)
                print("mutate_connectivity_scheme: " + self.mutate_connectivity_scheme, file=text_file)
                print("recombine_method: " + self.recombine_method.__name__, file=text_file)
                print("recombine_connectivity_scheme: " + self.recombine_connectivity_scheme, file=text_file)
                print("evaluate_population_method_for_selection: " +
                      self.evaluate_population_method_for_selection.__name__, file=text_file)
                print("evaluate_population_method_for_recombination: " +
                      self.evaluate_population_method_for_recombination.__name__, file=text_file)
                print("evaluate_individual_method: " + self.evaluate_individual_method.__name__, file=text_file)
                print("param_reinitialization_method: " + self.param_reinitialization_method.__name__, file=text_file)
                print("NN_architecture: " + str(get_network_architecture(model)), file=text_file)
                print("X_for_accuracy: " + self.X_for_accuracy[1], file=text_file)
                print("Y_for_accuracy: " + str(self.Y_for_accuracy[1]), file=text_file)

        # Initialize OptimizationProblem and start population
        if self.seed:
            # seed=0 doesn't work. Why?
            np.random.seed(self.seed)

        self.wtp = WinningTicketProblem(seed=self.seed, model=self.model,
                                        generate_method=self.generate_method, mutate_method=self.mutate_method,
                                        recombine_method=self.recombine_method, evaluate_population_method_for_selection
                                        =self.evaluate_population_method_for_selection,
                                        evaluate_population_method_for_recombination
                                        =self.evaluate_population_method_for_recombination,
                                        evaluate_individual_method=self.evaluate_individual_method,
                                        param_reinitialization_method=self.param_reinitialization_method,
                                        X_for_accuracy=self.X_for_accuracy[0], Y_for_accuracy=self.Y_for_accuracy[0])

    def start_population(self, id, NR_IND, mig_rate, mut_rate, rec_rate, par_rate, selection_method, selection_args,
                         evaluate_for_selection_args, evaluate_for_recombination_args, evaluate_individual_args,
                         mutate_args, recombine_args, generate_args):
        # Log population settings
        if self.log:
            with open(self.log_path + f"/population{id}_settings.txt", "w") as text_file:
                print("NR_IND: " + str(NR_IND), file=text_file)
                print("mig_rate: " + str(mig_rate), file=text_file)
                print("mut_rate: " + str(mut_rate), file=text_file)
                print("rec_rate: " + str(rec_rate), file=text_file)
                print("par_rate: " + str(par_rate), file=text_file)
                print("selection_method: " + selection_method.__name__, file=text_file)
                print("selection_args: " + str(selection_args), file=text_file)
                print("evaluate_for_selection_args: " + str(evaluate_for_selection_args), file=text_file)
                print("evaluate_for_recombination_args: " + str(evaluate_for_recombination_args), file=text_file)
                print("evaluate_individual_args: " + str(evaluate_individual_args), file=text_file)
                print("mutate_args: " + str(mutate_args), file=text_file)
                print("recombine_args: " + str(recombine_args), file=text_file)
                print("generate_args: " + str(generate_args), file=text_file)

        return Population(self.wtp, individuals=NR_IND, mig=mig_rate, mut=mut_rate,
                          rec=rec_rate, par=par_rate, selection=selection_method,
                          selection_args=selection_args, evaluate_for_selection_args
                          =evaluate_for_selection_args, evaluate_for_recombination_args
                          =evaluate_for_recombination_args, evaluate_individual_args
                          =evaluate_individual_args, mutate_args=mutate_args,
                          recombine_args=recombine_args, generate_args=generate_args)

    def evolute(self, population, initial_routines, evolution_routines, final_routines, termination_condition):
        tmp_log = {"log_path": self.log_path} if self.log else {}
        # Routines to fill tmp_log with entries that are important for later routines
        for routine in initial_routines:
            routine(population, tmp_log)

        while not termination_condition(population, tmp_log):
            population.evolve()

            for routine in evolution_routines:
                routine(population, tmp_log)

        for routine in final_routines:
            routine(population, tmp_log)


def choose_evolution_subroutines_with_dependencies(routines):
    ir_set, er_set, fr_set = set(), set(), set()
    termination_condition = None

    def choose_routine_of_type(routine_type, routine_number):
        if routine_type == "init":
            return init_routines_dict.get(routine_number)
        else:
            if routine_type == "evol":
                routine_dict = evol_routines_dict
            elif routine_type == "final":
                routine_dict = final_routines_dict
            else:
                routine_dict = termination_conditions_dict
            routine_with_dependencies = routine_dict.get(routine_number)
            routine = routine_with_dependencies[0]
            ir_dependencies = routine_with_dependencies[1]
            er_dependencies = routine_with_dependencies[2]
            return routine, ir_dependencies, er_dependencies

    for r in routines:
        routine_type = r[0]
        routine_number = r[1]

        if routine_type == "init":
            ir_set.add(choose_routine_of_type("init", routine_number))
        else:
            routine, ir_dependencies, er_dependencies = choose_routine_of_type(routine_type, routine_number)
            if routine_type == "evol":
                er_set.add(routine)
            elif routine_type == "final":
                fr_set.add(routine)
            else:
                termination_condition = routine
            for ir in ir_dependencies:
                ir_set.add(ir)
            for er in er_dependencies:
                er_set.add(er)

    initial_routines, evolution_routines, final_routines = list(ir_set), list(er_set), list(fr_set)
    initial_routines.sort(key=lambda routine: int(routine.__name__[2:]))
    evolution_routines.sort(key=lambda routine: int(routine.__name__[2:]))
    final_routines.sort(key=lambda routine: int(routine.__name__[2:]))

    return initial_routines, evolution_routines, final_routines, termination_condition
