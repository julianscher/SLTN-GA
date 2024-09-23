import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
import dill

from src.experiments.base.GA_configurations import top_speed, max_acc_bound, max_acc_bound_small, max_acc
from src.neural_network.NN import get_model
from src.neural_network.dataset import get_dataset
from src.neural_network.reinitialization_methods import reinitialize_parameters2
from src.experiments.base.experiment_framework import Experiment, choose_evolution_subroutines_with_dependencies
from src.genetic_algorithm.evaluate_methods import evaluate_individual_method1, evaluate_population_method2, \
    evaluate_population_method1
from src.genetic_algorithm.generation import generate_lax_connectivity_learned_accuracy_filter

sys.path.append(str(Path(__file__).parent))
results_rel_path = "resources/test_results/GA_performance_results"
results_path = Path(os.path.join(Path(__file__).parent.parent.parent.parent, results_rel_path))

# Experiment contents overview
NN_architectures_binary = [[2, 20, 2], [2, 75, 2], [2, 100, 2], [2, 50, 50, 2]]
NN_architectures_digits = [[64, 20, 10], [64, 75, 10], [64, 100, 10], [64, 50, 50, 10]]
NN_architectures_digits_binary = [[64, 20, 2], [64, 75, 2], [64, 100, 2], [64, 50, 50, 2]]
GA_configs = [top_speed, max_acc_bound, max_acc_bound_small]
data = ["make_moons", "make_circles", "load_digits", "load_digits_binary"]

if not os.path.isdir(results_path):
    os.mkdir(results_path)



job_tasks_binary = {"tasks0": {i: [2, 20, 2] for i in range(50)},
                    "tasks1": {i: [2, 75, 2] for i in range(50)},
                    "tasks2": {i: [2, 100, 2] for i in range(50)},
                    "tasks3": {i: [2, 50, 50, 2] for i in range(50)}}

job_tasks_digits = {"tasks0": {i: [64, 20, 10] for i in range(50)},
                    "tasks1": {i: [64, 75, 10] for i in range(50)},
                    "tasks2": {i: [64, 100, 10] for i in range(50)},
                    "tasks3": {i: [64, 50, 50, 10] for i in range(50)}}

job_tasks_digits_binary = {"tasks0": {i: [64, 20, 2] for i in range(50)},
                           "tasks1": {i: [64, 75, 2] for i in range(50)},
                           "tasks2": {i: [64, 100, 2] for i in range(50)},
                           "tasks3": {i: [64, 50, 50, 2] for i in range(50)}}


def run_job_with_checkpointing(job_id, selected_tasks, tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test,
                               reinitialization_method, test_for_speed, experiment_directory):
    # Get start point considering potential restarts
    continue_at = check_if_job_was_restarted(job_id, selected_tasks[0], experiment_directory)

    for (counter, NN_architecture) in {k: tasks.get(k, None) for k in
                                       selected_tasks[selected_tasks.index(continue_at):]}.items():

        # Store task to start at after job restart
        with open(f"{results_path}/{experiment_directory}/interrupt_logs/interrupt_job{job_id}.txt", 'w') as text_file:
            print(counter, file=text_file)

        if os.path.isdir(f"{results_path}/{experiment_directory}/job_{job_id}_{counter}/"):
            # If previous execution of this task was unsuccessful, remove past files and rerun task, else skip
            if not check_for_unfinished_evolution(job_id, counter, experiment_directory):
                continue

        if GA_config.get("generate_method") is generate_lax_connectivity_learned_accuracy_filter:
            if (len(NN_architecture) == 3 and NN_architecture[1] < 40) or (len(NN_architecture) == 4 and NN_architecture[1] < 30 and NN_architecture[2] < 30):
                GA_config = max_acc_bound_small

        (generate_method, mutate_method, recombine_method, selection_method, selection_args, generate_args,
         mutate_args, recombine_args) = GA_config.values()

        run_job_with_timeout(f"{experiment_directory}/job_{job_id}_{counter}", NN_architecture,
                             mutate_method, mutate_args, generate_method, generate_args, selection_method,
                             recombine_method, recombine_args, reinitialization_method, X_train_named,
                             Y_train_named, X_test, Y_test, test_for_speed, 0)

def run_job_with_timeout(name, NN_architecture, mutate_method,
                         mutate_args, generate_method, generate_args, selection_method, recombine_method,
                         recombine_args, reinitialization_method, X_train_named, Y_train_named,
                         X_test, Y_test, test_for_speed, seed):
    """ This method only works with Unix systems """

    """def handler(signum, frame):
        print("Forever is over!")
        raise Exception("end of time")

    signal.signal(signal.SIGALRM, handler)
    # Give every task 1800 sec to finish
    signal.alarm(1800)"""

    try:
        start_time = time.time()
        print(f"Starting experiment {name}", start_time)
        exp = Experiment(name=f"GA_performance_results/{name}", seed=seed,
                         generate_method=generate_method,
                         mutate_method=mutate_method,
                         mutate_connectivity_scheme=mutate_args.get("connectivity_scheme"),
                         recombine_method=recombine_method,
                         recombine_connectivity_scheme=recombine_args.get("connectivity_scheme"),
                         evaluate_population_method_for_selection=evaluate_population_method1,
                         evaluate_population_method_for_recombination=evaluate_population_method2,
                         evaluate_individual_method=evaluate_individual_method1,
                         param_reinitialization_method=reinitialization_method,
                         model=get_model(NN_architecture, seed, reinitialization_method),
                         X_for_accuracy=(X_train_named[0], X_train_named[1]),
                         Y_for_accuracy=(Y_train_named[0], Y_train_named[1]),
                         log=True)

        start_population = exp.start_population(id=1, mig_rate=0.1, mut_rate=0.1, rec_rate=0.3,
                                                par_rate=0.3,
                                                selection_method=selection_method,
                                                selection_args={}, generate_args=generate_args,
                                                mutate_args=mutate_args,
                                                recombine_args=recombine_args,
                                                evaluate_for_selection_args={},
                                                evaluate_for_recombination_args={},
                                                evaluate_individual_args={}, NR_IND=100)

        try:
            if test_for_speed:
                routines = [("evol", 8), ("final", 11), ("final", 13), ("term", 4 if generate_method is generate_lax_connectivity_learned_accuracy_filter else 10)]
            else:
                routines = [("init", 13), ("evol", 1), ("evol", 2), ("evol", 6), ("evol", 7), ("evol", 8), ("evol", 9),
                            ("evol", 10), ("evol", 13), ("evol", 14), ("evol", 15), ("final", 11), ("final", 13),
                            ("term", 4 if generate_method is generate_lax_connectivity_learned_accuracy_filter else 10)]
            initial_routines, evolution_routines, final_routines, termination_condition \
                = choose_evolution_subroutines_with_dependencies(routines)
            exp.evolute(start_population, initial_routines, evolution_routines, final_routines,
                        termination_condition)

            # Evaluate final population on test dataset
            start_population.evaluate_accuracy_on_test_dataset(X_test, Y_test)
            final_evaluated_population_flattened = {}
            for rank, individual in enumerate(start_population.individuals):
                final_evaluated_population_flattened[rank] = {"genome": individual.genome, "origin": individual.origin,
                                                              "fitness": individual.fitness}
            with open(exp.log_path + "/pickled_objects/pickled_final_evaluated_population_flattened", "wb") as dill_file:
                dill.dump(final_evaluated_population_flattened, dill_file)

            runtime = time.time() - start_time
            print("Time to run: {} seconds".format(runtime))
            store_runtime(runtime, exp.log_path)
        except Exception:
            with open(exp.log_path + "/error_log.txt", "w") as text_file:
                print(traceback.format_exc(), file=text_file)

            runtime = 1800
            store_runtime(runtime, exp.log_path)
    except Exception:
        print(traceback.print_exc())


##### helper functions #####

def check_if_job_was_restarted(job_id, start_at, experiment_directory):
    interrupt_job = open(f"{results_path}/{experiment_directory}/interrupt_logs/interrupt_job{job_id}.txt", 'r')
    continue_at = interrupt_job.readline()
    if continue_at:
        continue_at = int(continue_at)
    else:
        continue_at = start_at

    return continue_at


def check_for_unfinished_evolution(job_id, task, experiment_directory):
    is_unfinished_evolution = False
    # remove directory for unfinished evolution
    if not os.path.isfile(f"{results_path}/{experiment_directory}/job_{job_id}_{task}/pickled_objects/pickled_tmp_log"):
        try:
            shutil.rmtree(f"{results_path}/{experiment_directory}/job_{job_id}_{task}")
            is_unfinished_evolution = True
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            sys.exit()

    return is_unfinished_evolution


def store_runtime(runtime, path):
    if not os.path.isdir(f"{path}" + "/pickled_objects"):
        os.mkdir(f"{path}" + "/pickled_objects")
    with open(f"{path}" + "/pickled_objects/pickled_runtime", "wb") as dill_file:
        dill.dump(runtime, dill_file)


def create_experiment_directory_with_interrupt_logs(target_directory, job_count):
    target_directory_components = target_directory.split("/")
    if not os.path.isdir(f"{results_path}/{target_directory}"):
        for idx, directory_component in enumerate(target_directory_components):
            if not os.path.isdir(f"{results_path}/{'/'.join(target_directory_components[:idx+1])}"):
                os.mkdir(f"{results_path}/{'/'.join(target_directory_components[:idx+1])}")
        os.mkdir(f"{results_path}/{target_directory}/interrupt_logs")
        # create interrupt logs
        for i in range(job_count):
            open(f"{results_path}/{target_directory}/interrupt_logs/interrupt_job{i}.txt", "x")


def clear_interrupt_log(job_id, experiment_directory):
    open(f"{results_path}/{experiment_directory}/interrupt_logs/interrupt_job{job_id}.txt", 'w').close()


##### conduction function #####

def run_tasks_job(job_id, selected_tasks, job_tasks, GA_config, X_train_named,
                  Y_train_named, X_test, Y_test, reinitialization_method, test_for_speed, experiment_directory):
    clear_interrupt_log(job_id, experiment_directory)
    run_job_with_checkpointing(job_id, selected_tasks, job_tasks, GA_config, X_train_named,
                               Y_train_named, X_test, Y_test, reinitialization_method, test_for_speed, experiment_directory)


##### Experiment settings #####
def get_experiment1():
    # You have to create the directory make_moons manually in GA_performance_results
    experiment_directory = "make_moons/GA_evolutions_top_speed"
    job_tasks = job_tasks_binary
    GA_config = top_speed
    X_train_named = (get_dataset("make_moons")["X_train_std"], "X_train_make_moons(n_samples=50000, random_state=42, noise=0.07)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset("make_moons")["Y_train"], "Y_train_make_moons(n_samples=50000, random_state=42, noise=0.07)")
    X_test = get_dataset("make_moons")["X_test_std"]
    Y_test = get_dataset("make_moons")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method


def get_experiment2():
    # You have to create the directory make_moons manually in GA_performance_results
    experiment_directory = "make_moons/GA_evolutions_max_acc_bound"
    job_tasks = job_tasks_binary
    GA_config = max_acc_bound
    X_train_named = (get_dataset("make_moons")["X_train_std"], "X_train_make_moons(n_samples=50000, random_state=42, noise=0.07)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset("make_moons")["Y_train"], "Y_train_make_moons(n_samples=50000, random_state=42, noise=0.07)")
    X_test = get_dataset("make_moons")["X_test_std"]
    Y_test = get_dataset("make_moons")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method


def get_experiment3():
    # You have to create the directory make_circles manually in GA_performance_results
    experiment_directory = "make_circles/GA_evolutions_top_speed"
    job_tasks = job_tasks_binary
    GA_config = top_speed
    X_train_named = (get_dataset("make_circles")["X_train"], "X_train_make_circles(n_samples=50000, random_state=42, noise=0.07)")
    Y_train_named = (get_dataset("make_circles")["Y_train"], "Y_train_make_circles(n_samples=50000, random_state=42, noise=0.07)")
    X_test = get_dataset("make_circles")["X_test"]
    Y_test = get_dataset("make_circles")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method


def get_experiment4():
    # You have to create the directory make_circles manually in GA_performance_results
    experiment_directory = "make_circles/GA_evolutions_max_acc_bound"
    job_tasks = job_tasks_binary
    GA_config = max_acc_bound
    X_train_named = (get_dataset("make_circles")["X_train"], "X_train_make_circles(n_samples=50000, random_state=42, noise=0.07)")
    Y_train_named = (get_dataset("make_circles")["Y_train"], "Y_train_make_circles(n_samples=50000, random_state=42, noise=0.07)")
    X_test = get_dataset("make_circles")["X_test"]
    Y_test = get_dataset("make_circles")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method


def get_experiment5():
    # You have to create the directory load_digits manually in GA_performance_results
    experiment_directory = "load_digits/GA_evolutions_top_speed"
    job_tasks = job_tasks_digits
    GA_config = top_speed
    X_train_named = (get_dataset("load_digits")["X_train_std"], "X_train_load_digits(test_size=0.25)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset("load_digits")["Y_train"], "Y_train_load_digits(test_size=0.25)")
    X_test = get_dataset("load_digits")["X_test_std"]
    Y_test = get_dataset("load_digits")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method

def get_experiment6():
    # You have to create the directory load_digits manually in GA_performance_results
    experiment_directory = "load_digits/GA_evolutions_max_acc"
    job_tasks = job_tasks_digits
    GA_config = max_acc
    GA_config["generate_args"] = {"bound": 0.15}
    X_train_named = (get_dataset("load_digits")["X_train_std"], "X_train_load_digits(test_size=0.25)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset("load_digits")["Y_train"], "Y_train_load_digits(test_size=0.25)")
    X_test = get_dataset("load_digits")["X_test_std"]
    Y_test = get_dataset("load_digits")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method

def get_experiment7():
    # You have to create the directory load_digits_binary manually in GA_performance_results
    experiment_directory = "load_digits_binary/GA_evolutions_top_speed"
    job_tasks = job_tasks_digits_binary
    GA_config = top_speed
    X_train_named = (get_dataset("load_digits_binary")["X_train_std"], "X_train_load_digits_binary(test_size=0.25)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset("load_digits_binary")["Y_train"], "Y_train_load_digits_binary(test_size=0.25)")
    X_test = get_dataset("load_digits_binary")["X_test_std"]
    Y_test = get_dataset("load_digits_binary")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method

def get_experiment8():
    # You have to create the directory load_digits_binary manually in GA_performance_results
    experiment_directory = "load_digits_binary/GA_evolutions_max_acc_bound"
    job_tasks = job_tasks_digits_binary
    GA_config = max_acc_bound
    X_train_named = (get_dataset("load_digits_binary")["X_train_std"], "X_train_load_digits_binary(test_size=0.25)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset("load_digits_binary")["Y_train"], "Y_train_load_digits_binary(test_size=0.25)")
    X_test = get_dataset("load_digits_binary")["X_test_std"]
    Y_test = get_dataset("load_digits_binary")["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method

def get_digits_experiment(n_classes):
    # You have to create the directory load_digits_fewer_classes manually in GA_performance_results
    digits_datasets = {2: "load_digits_binary", 3: "load_digits_ternary", 4: "load_digits_quaternary", 5: "load_digits_quinary"}
    selected_dataset = digits_datasets[n_classes]
    experiment_directory = f"load_digits_fewer_classes/{selected_dataset}"
    job_tasks = {"tasks0": {i: [64, 75, n_classes] for i in range(50)}}
    GA_config = top_speed
    X_train_named = (get_dataset(selected_dataset)["X_train_std"], f"X_train_{selected_dataset}(test_size=0.25)_std(-0.7, 0.7)")
    Y_train_named = (get_dataset(selected_dataset)["Y_train"], f"Y_train_{selected_dataset}(test_size=0.25)")
    X_test = get_dataset(selected_dataset)["X_test_std"]
    Y_test = get_dataset(selected_dataset)["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method


def get_blob_experiment(n_classes):
    # You have to create the directory make_blobs manually in GA_performance_results
    blob_datasets = {2: "make_two_blobs", 3: "make_three_blobs", 4: "make_four_blobs", 5: "make_five_blobs", 6: "make_six_blobs",
                     7: "make_seven_blobs", 8: "make_eight_blobs", 9: "make_nine_blobs", 10: "make_ten_blobs"}
    selected_dataset = blob_datasets[n_classes]
    experiment_directory = f"make_blobs/{selected_dataset}"
    job_tasks = {"tasks0": {i: [2, 100, n_classes] for i in range(50)}}
    GA_config = top_speed
    X_train_named = (get_dataset(selected_dataset)["X_train_std"], f"X_train_{selected_dataset}(test_size=0.25)")
    Y_train_named = (get_dataset(selected_dataset)["Y_train"], f"Y_train_{selected_dataset}(test_size=0.25)")
    X_test = get_dataset(selected_dataset)["X_test_std"]
    Y_test = get_dataset(selected_dataset)["Y_test"]
    reinitialization_method = reinitialize_parameters2
    return experiment_directory, job_tasks, GA_config, X_train_named, Y_train_named, X_test, Y_test, reinitialization_method


if __name__ == "__main__":
    sys.setrecursionlimit(50000) # To prevent RecursionError
    experiment_directory, job_tasks, GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method = get_experiment1()
    create_experiment_directory_with_interrupt_logs(experiment_directory, 4)
    run_tasks_job(0, range(1), job_tasks["tasks0"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(0, range(25, 50), job_tasks["tasks0"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(1, range(25), job_tasks["tasks1"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(1, range(25, 50), job_tasks["tasks1"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(2, range(25), job_tasks["tasks2"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(2, range(25, 50), job_tasks["tasks2"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(3, range(25), job_tasks["tasks3"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
    # run_tasks_job(3, range(25, 50), job_tasks["tasks3"], GA_config, X_test_named, Y_test_named, X_test, Y_test, reinitialization_method, False, experiment_directory)
