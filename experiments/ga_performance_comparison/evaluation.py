import copy
import os
import re

import dill
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from scipy import stats

from data import DIGITS, BLOBS
from experiments.backprop_performance_comparison.evaluation import get_accuracy_data, \
    get_median_accuracy_for_architecture, get_mean_accuracy_for_architecture, \
    get_make_blobs_load_digits_fewer_classes_data_backprop
from models.subnetworks.subnetworks import SubnetMLP
from utilities.helper_functions import get_results_path
from utilities.net_utils import accuracy

results_path = f"{get_results_path()}/ga_performance_results"

test_runs = [name for name in os.listdir(results_path) if name[:4] == "job_"]
all_test_runs = [f"job_{job_id}_{task}" for job_id in [0, 1, 2, 3] for task in [i for i in range(50)]]
config_name = ["GA_evolutions_top_speed", "GA_evolutions_max_acc_bound", "GA_evolutions_max_acc"]
dataset_names = ["make_moons", "make_circles", "load_digits", "load_digits_binary"]
NN_architectures_binary = ["[2, 20, 2]", "[2, 75, 2]", "[2, 100, 2]", "[2, 50, 50, 2]"]
NN_architectures_digits = ["[64, 20, 10]", "[64, 75, 10]", "[64, 100, 10]", "[64, 50, 50, 10]"]
NN_architectures_digits_binary = ["[64, 20, 2]", "[64, 75, 2]", "[64, 100, 2]", "[64, 50, 50, 2]"]


def get_NN_architectures(dataset_name):
    if dataset_name in ["make_moons", "make_circles"]:
        return NN_architectures_binary
    elif dataset_name == "load_digits":
        return NN_architectures_digits
    else:
        return NN_architectures_digits_binary

def get_list_of_unsuccessful_runs(dataset_name, config_name, file_type):
    """ file_type: ["final_population", "initial_population", "runtime", "tmp_log"] """
    config_path = f"{results_path}/{dataset_name}/{config_name}"
    test_runs = [name for name in os.listdir(f"{results_path}/{dataset_name}/{config_name}/") if name[:4] == "job_"]
    unsuccessful_runs = []
    for test_run in test_runs:
        try:
            with open(f"{config_path}/{test_run}/pickled_objects/pickled_{file_type}", 'rb') as pickle_file:
                dill.load(pickle_file)
        except (EOFError, FileNotFoundError):
            unsuccessful_runs.append(test_run)
            continue
    return list(set(unsuccessful_runs))


def get_list_of_missing_runs(dataset_name, config_name):
    missing_runs = []
    for test_run in all_test_runs:
        if not os.path.isdir(f"{results_path}/{dataset_name}/{config_name}/{test_run}/"):
            missing_runs.append(test_run)
    return missing_runs


def get_make_moons_make_circles_load_digits_data(dataset_name, config_name, metric_method, training_accuracy):
    config_path = f"{results_path}/{dataset_name}/{config_name}"
    test_runs = [name for name in os.listdir(f"{results_path}/{dataset_name}/{config_name}/") if name[:4] == "job_"]
    NN_architectures = get_NN_architectures(dataset_name)
    data = {str(NN_architecture): [] for NN_architecture in NN_architectures}
    print(data.keys())
    origins = {"tmp_log":0, "pickled_final_evaluated_population": 0, "pickled_final_evaluated_population_flattened": 0, "unsuccessful_run": []}
    for test_run in [test_run for test_run in test_runs]:
        text_file = open(f"{config_path}/{test_run}/wtp_settings.txt", 'rb')
        wtp_settings = text_file.readlines()
        NN_architecture = [str(setting)[str(setting).index(":"):][2:-3] for setting in wtp_settings
                           if re.findall("\'.*?:", str(setting)) and re.findall("\'.*?:", str(setting))[0][1:-1] == "NN_architecture"][0]

        # remove potential white spaces from pattern
        NN_architecture = NN_architecture[:NN_architecture.index("]") + 1]

        metric, origin = metric_method(training_accuracy, config_path, test_run)

        if origin == "unsuccessful_run":
            origins["unsuccessful_run"].append(test_run)
        else:
            origins[origin] += 1
        if not metric:
            continue

        data[NN_architecture].append(metric)

    # Sanity check
    print(origins)
    for key in data.keys():
        print(f"{key} valid runs:", len(data[key]))
    return data


def get_make_blobs_load_digits_fewer_classes_data(dataset_name, metric_method, training_accuracy, evaluate=None, source="tmp_log"):
    dataset_path = f"{results_path}/{dataset_name}/"
    if "load_digits_fewer_classes" in dataset_name:
        dataset_instances = ["load_digits_binary", "load_digits_ternary", "load_digits_quaternary", "load_digits_quinary", "load_digits_denary"]
    else:
        dataset_instances = ["make_two_blobs", "make_three_blobs", "make_four_blobs", "make_five_blobs", "make_six_blobs", "make_seven_blobs", "make_eight_blobs", "make_nine_blobs", "make_ten_blobs"]
    test_runs = {dataset_instance: [name for name in os.listdir(f"{results_path}/{dataset_name}/{dataset_instance}/") if name[:4] == "job_"] for dataset_instance in dataset_instances}
    data = {dataset_instance: [] for dataset_instance in dataset_instances}
    print(data.keys())
    origins = {"tmp_log": 0, "pickled_final_evaluated_population": 0, "pickled_final_evaluated_population_flattened": 0, "unsuccessful_run": []}
    for dataset_instance in dataset_instances:
        for test_run in [test_run for test_run in test_runs[dataset_instance]]:
            dataset_instance_path = f"{dataset_path}/{dataset_instance}/"
            if evaluate:
                metric, origin = evaluate_top_individual(training_accuracy, dataset_instance_path, test_run, evaluate, source)
            else:
                metric, origin = metric_method(training_accuracy, dataset_instance_path, test_run)

            if origin == "unsuccessful_run":
                origins["unsuccessful_run"].append(test_run)
            else:
                origins[origin] += 1
            if not metric:
                continue

            data[dataset_instance].append(metric)

    # Sanity check
    print(origins)
    for key in data.keys():
        print(f"{key} valid runs:", len(data[key]))
    return data

def get_top_accuracy(training_accuracy, path, test_run):
    try:
        if training_accuracy:
            with open(f"{path}/{test_run}/pickled_objects/pickled_tmp_log", 'rb') as pickle_file:
                tmp_log = dill.load(pickle_file)
            if isinstance(tmp_log["accuracy_development"], list):
                return max(tmp_log["accuracy_development"]), "tmp_log"
            elif isinstance(tmp_log["accuracy_development"], dict):
                return max(tmp_log["accuracy_development"].values()), "tmp_log"
        else:
            if os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population"):
                with open(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population",
                          'rb') as pickle_file:
                    final_evaluated_population = dill.load(pickle_file)
                return final_evaluated_population.individuals[0].fitness, "pickled_final_evaluated_population"

            elif os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population_flattened"):
                with open(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population_flattened",
                          'rb') as pickle_file:
                    final_evaluated_population_flattened = dill.load(pickle_file)
                return final_evaluated_population_flattened[0].get("fitness"), "pickled_final_evaluated_population_flattened"
            else:
                raise ValueError(f"There is no evaluated population for {test_run}")
    except (EOFError, FileNotFoundError, ValueError):
        return None, "unsuccessful_run"


def evaluate_top_individual(training_accuracy, path, test_run, evaluate, source):
    nn_architecture = copy.deepcopy(evaluate["model_args"]["nn_architecture"])
    if "load_digits" in path:
        if "_binary" in path:
            nn_architecture.append(2)
            if evaluate["dataset_args"]["name"] == "digits":
                data = DIGITS(evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"], [0, 1])
        elif "_ternary" in path:
            nn_architecture.append(3)
            if evaluate["dataset_args"]["name"] == "digits":
                data = DIGITS(evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"], [0, 1, 2])
        elif "_quaternary" in path:
            nn_architecture.append(4)
            if evaluate["dataset_args"]["name"] == "digits":
                data = DIGITS(evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"], [0, 1, 2, 3])
        elif "_quinary" in path:
            nn_architecture.append(5)
            if evaluate["dataset_args"]["name"] == "digits":
                data = DIGITS(evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"], [0, 1, 2, 3, 4])
        elif "_denary" in path:
            nn_architecture.append(10)
            if evaluate["dataset_args"]["name"] == "digits":
                data = DIGITS(evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"],
                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    elif "make_blobs" in path:
        if "_two" in path:
            nn_architecture.append(2)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(2, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_three" in path:
            nn_architecture.append(3)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(3, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_four" in path:
            nn_architecture.append(4)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(4, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_five" in path:
            nn_architecture.append(5)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(5, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_six" in path:
            nn_architecture.append(6)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(6, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_seven" in path:
            nn_architecture.append(7)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(7, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_eight" in path:
            nn_architecture.append(8)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(8, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_nine" in path:
            nn_architecture.append(9)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(9, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])
        elif "_ten" in path:
            nn_architecture.append(10)
            if evaluate["dataset_args"]["name"] == "blobs":
                data = BLOBS(10, evaluate["dataset_args"]["norm"], evaluate["dataset_args"]["device"])

    model = SubnetMLP(nn_architecture, evaluate["model_args"]["init"], evaluate["model_args"]["prune_rate"],
                      evaluate["model_args"]["activation"], evaluate["model_args"]["mask_except"],
                      evaluate["model_args"]["device"])
    model.net.load_state_dict(torch.load(f"{path}/{test_run}/model_init.pt"))

    if training_accuracy:
        loader = data.train_loader
    else:
        loader = data.test_loader

    if source == "tmp_log" and os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_tmp_log"):
        with open(f"{path}/{test_run}/pickled_objects/pickled_tmp_log", 'rb') as pickle_file:
            tmp_log = dill.load(pickle_file)
            best_individual = tmp_log["max_pruned_SLT"]["bit_vector"]
            origin = "tmp_log"
    elif source == "pickled_final_population_flattened" and os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_final_population_flattened"):
        with open(f"{path}/{test_run}/pickled_objects/pickled_final_population_flattened",
                  'rb') as pickle_file:
            final_population_flattened = dill.load(pickle_file)

        best_individual = None
        if f"{path}/{test_run}" == f"{results_path}/ga_performance_results/make_blobs//make_five_blobs//job_0":
            print("Evaluate population")
        for individual in final_population_flattened.values():
            mask = individual["genome"]
            masked_model = model.apply_mask(mask)

            X = loader.dataset.tensors[0]
            y = loader.dataset.tensors[1]

            if f"{path}/{test_run}" == f"{results_path}/ga_performance_results/make_blobs//make_five_blobs//job_0":
                print("loss:", individual["fitness"])
            individual["fitness"] = accuracy(masked_model(X), y, evaluate["dataset_args"]["device"]) / 100
            if f"{path}/{test_run}" == f"{results_path}/ga_performance_results/make_blobs//make_five_blobs//job_0":
                print("accuracy: ", individual["fitness"])

            if not best_individual:
                best_individual = individual
            else:
                if individual["fitness"] > best_individual["fitness"]:
                    best_individual = individual

        return best_individual["fitness"], "pickled_final_evaluated_population"

    elif source == "pickled_final_evaluated_population_flattened" and os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population_flattened"):
        with open(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population_flattened",
                  'rb') as pickle_file:
            final_evaluated_population_flattened = dill.load(pickle_file)
        best_individual = final_evaluated_population_flattened[0]["genome"]
        origin = "pickled_final_evaluated_population_flattened"

    subnet = model.apply_mask(best_individual)

    subnet.eval()
    total_metric = 0
    X = loader.dataset.tensors[0]
    y = loader.dataset.tensors[1]
    total_metric = accuracy(subnet(X), y, evaluate["dataset_args"]["device"]) / 100
    """with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(evaluate["dataset_args"]["device"]), y_batch.to(evaluate["dataset_args"]["device"])
            output = subnet(X_batch)
            total_metric += accuracy(output, y_batch, evaluate["dataset_args"]["device"])
    total_metric = total_metric / len(loader)"""
    return total_metric, origin


def get_top_accuracy_individuals_sparsity(training_accuracy, path, test_run):
    try:
        if training_accuracy:
            with open(f"{path}/{test_run}/pickled_objects/pickled_tmp_log", 'rb') as pickle_file:
                tmp_log = dill.load(pickle_file)
            return tmp_log["max_0_count_of_highest_accuracy_development"][-1], "tmp_log"
        else:
            if os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population"):
                with open(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population",
                          'rb') as pickle_file:
                    final_evaluated_population = dill.load(pickle_file)
                return final_evaluated_population.individuals[0].genome.count(0), "pickled_final_evaluated_population"

            elif os.path.isfile(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population_flattened"):
                with open(f"{path}/{test_run}/pickled_objects/pickled_final_evaluated_population_flattened",
                          'rb') as pickle_file:
                    final_evaluated_population_flattened = dill.load(pickle_file)
                return final_evaluated_population_flattened[0].get("genome").count(0), "pickled_final_evaluated_population_flattened"
            else:
                raise ValueError(f"There is no evaluated population for {test_run}")
    except (EOFError, FileNotFoundError, ValueError):
        return None, "unsuccessful_run"


def get_mean_runtime(dataset_name, config_name):
    test_runs = [name for name in os.listdir(f"{results_path}/{dataset_name}/{config_name}/") if name[:4] == "job_"]

    runtimes = {"[2, 20, 2]": [], "[2, 75, 2]": [], "[2, 100, 2]": [], "[2, 50, 50, 2]": []}
    for test_run in test_runs:
        try:
            with open(f"{results_path}/{dataset_name}/{config_name}/{test_run}/pickled_objects/pickled_runtime", 'rb') as dill_file:
                runtime = dill.load(dill_file)
        except:
            continue
        if test_run[4] == "0":
            runtimes["[2, 20, 2]"].append(runtime)
        elif test_run[4] == "1":
            runtimes["[2, 75, 2]"].append(runtime)
        elif test_run[4] == "2":
            runtimes["[2, 100, 2]"].append(runtime)
        elif test_run[4] == "3":
            runtimes["[2, 50, 50, 2]"].append(runtime)

    return {key: np.mean(runtimes_) for key, runtimes_ in runtimes.items()}

def get_mean_metric_value_make_moons_make_circles_load_digits(dataset_name, config_name, metric_method, training_accuracy):
    data = get_make_moons_make_circles_load_digits_data(dataset_name, config_name, metric_method, training_accuracy)
    mean_metric_value = {NN_architecture: 0 for NN_architecture in data.keys()}
    for key in data.keys():
        mean_metric_value[key] = np.mean(data[key])
    return mean_metric_value


def get_median_metric_value_make_moons_make_circles_load_digits(dataset_name, config_name, metric_method, training_accuracy):
    data = get_make_moons_make_circles_load_digits_data(dataset_name, config_name, metric_method, training_accuracy)
    median_metric_value = {NN_architecture: 0 for NN_architecture in data.keys()}
    for key in data.keys():
        median_metric_value[key] = np.median(data[key])
    return median_metric_value

def get_and_store_make_moons_make_circles_mean_median_std(config_number, metric_method_name="get_top_accuracy"):
    metric_method = get_top_accuracy if metric_method_name == "get_top_accuracy" else get_top_accuracy_individuals_sparsity
    make_moons_data = get_make_moons_make_circles_load_digits_data("make_moons", config_name[config_number],
                                                                   metric_method, False)
    make_moons_data_2_20_2 = make_moons_data["[2, 20, 2]"]
    make_moons_data_2_75_2 = make_moons_data["[2, 75, 2]"]
    make_moons_data_2_100_2 = make_moons_data["[2, 100, 2]"]
    make_moons_data_2_50_50_2 = make_moons_data["[2, 50, 50, 2]"]

    make_circles_data = get_make_moons_make_circles_load_digits_data("make_circles", config_name[config_number],
                                                                     metric_method, False)
    make_circles_data_2_20_2 = make_circles_data["[2, 20, 2]"]
    make_circles_data_2_75_2 = make_circles_data["[2, 75, 2]"]
    make_circles_data_2_100_2 = make_circles_data["[2, 100, 2]"]
    make_circles_data_2_50_50_2 = make_circles_data["[2, 50, 50, 2]"]

    data = {
        "moons mean": [
            np.mean(make_moons_data_2_20_2),
            np.mean(make_moons_data_2_75_2),
            np.mean(make_moons_data_2_100_2),
            np.mean(make_moons_data_2_50_50_2)
        ],
        "moons median": [
            np.median(make_moons_data_2_20_2),
            np.median(make_moons_data_2_75_2),
            np.median(make_moons_data_2_100_2),
            np.median(make_moons_data_2_50_50_2)
        ],
        "moons std": [
            np.std(make_moons_data_2_20_2),
            np.std(make_moons_data_2_75_2),
            np.std(make_moons_data_2_100_2),
            np.std(make_moons_data_2_50_50_2)
        ],
        "circles mean": [
            np.mean(make_circles_data_2_20_2),
            np.mean(make_circles_data_2_75_2),
            np.mean(make_circles_data_2_100_2),
            np.mean(make_circles_data_2_50_50_2)
        ],
        "circles median": [
            np.median(make_circles_data_2_20_2),
            np.median(make_circles_data_2_75_2),
            np.median(make_circles_data_2_100_2),
            np.median(make_circles_data_2_50_50_2)
        ],
        "circles std": [
            np.std(make_circles_data_2_20_2),
            np.std(make_circles_data_2_75_2),
            np.std(make_circles_data_2_100_2),
            np.std(make_circles_data_2_50_50_2)
        ]
    }

    index_labels = ["2_20_2", "2_75_2", "2_100_2", "2_50_50_2"]
    df = pd.DataFrame(data, index=index_labels)

    excel_path = f"{results_path}/statistics/moons_circles_mean_median_std_for_{config_name[config_number]}_{metric_method_name}.xlsx"
    df.to_excel(excel_path)

    print(f"Results saved to {excel_path}")


def calculate_statistics_for_load_digits_fewer_classes_and_make_blobs_dataset(dataset_name, get_top_accuracy, training_accuracy,  evaluate=None, source="tmp_log", data=None):
    data = get_make_blobs_load_digits_fewer_classes_data(dataset_name, get_top_accuracy, training_accuracy, evaluate, source) if not data else data

    stats_dict = {}
    for key, values in data.items():
        values = np.array(values)
        min = values.min()
        max = values.max()
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values, ddof=1)
        if std == 0.0:
            ci_lower, ci_upper = mean, mean  # Set CI to mean if std is zero
        else:
            confidence_interval = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
            ci_lower, ci_upper = confidence_interval

        stats_dict[key] = [min, max, mean, median, std, ci_lower, ci_upper]

    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index',
                                      columns=['Min', 'Max', 'Mean', 'Median', 'Std Dev', '95% CI Lower', '95% CI Upper'])

    stats_df.to_csv(f"{results_path}/statistics/{dataset_name}_statistics.csv", index_label='Dataset')

    print("CSV file 'dataset_statistics.csv' has been created.")

def create_accuracy_box_plot_make_moons_make_circles_load_digits(dataset_name, config_name, training_accuracy):
    data = get_make_moons_make_circles_load_digits_data(dataset_name, config_name,  get_top_accuracy, training_accuracy)
    accuracies = list(data.values())
    NN_architectures = ["A", "B", "C", "D"]
    backprop_data = get_accuracy_data(dataset_name)
    backprop_median_accuracies = [get_median_accuracy_for_architecture(NN_architecture, dataset_name) for NN_architecture in backprop_data.keys()]

    fig = plt.figure(figsize=(9, 8))

    ax = fig.add_subplot(111)

    ax.set_xticklabels(NN_architectures)
    ax.scatter(range(1, len(backprop_median_accuracies) + 1), backprop_median_accuracies, color='red', label='Backpropagation Median', marker='o', linewidths=3)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    bp = ax.boxplot(accuracies, patch_artist=True)

    plt.xlabel("Architectures", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{results_path}/plots/GA_performance_comparison_boxplot_{dataset_name}_{config_name}_{'training_accuracy' if training_accuracy else 'testing_accuracy'}.png")


def create_accuracy_box_plot_make_moons_make_circles_load_digits_both_configs(dataset_name, box_colors, training_accuracy, grid=False, backprop_median=False):
    top_speed_data = get_make_moons_make_circles_load_digits_data(dataset_name, config_name[0], get_top_accuracy, training_accuracy)
    max_acc_data = get_make_moons_make_circles_load_digits_data(dataset_name, config_name[2] if dataset_name == "load_digits" else config_name[1], get_top_accuracy, training_accuracy)
    top_speed_accuracies = list(top_speed_data.values())
    max_acc_accuracies = list(max_acc_data.values())
    accuracies = []
    for i in range(len(top_speed_accuracies)):
        accuracies.append(top_speed_accuracies[i])
        accuracies.append(max_acc_accuracies[i])

    NN_architectures = ["A", "B", "C", "D"]
    backprop_data = get_accuracy_data(dataset_name)
    backprop_central_tendency = get_median_accuracy_for_architecture if backprop_median else get_mean_accuracy_for_architecture
    backprop_central_tendency_accuracies = [backprop_central_tendency(NN_architecture, dataset_name) for NN_architecture in backprop_data.keys()]
    backprop_central_tendency_accuracies_doubled = []
    for backprop_central_tendency_accuracy in backprop_central_tendency_accuracies:
        backprop_central_tendency_accuracies_doubled.extend([backprop_central_tendency_accuracy, backprop_central_tendency_accuracy])

    accuracies_in_percent = []
    for acc in accuracies:
        acc_in_percentage = []
        for i in acc:
            acc_in_percentage.append(i*100)
        accuracies_in_percent.append(acc_in_percentage)
    backprop_central_tendency_accuracies_doubled_in_percent = [acc*100 for acc in backprop_central_tendency_accuracies_doubled]

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))

    ax = fig.add_subplot(111)

    # Add backpropagation comparison plot
    ax.step(range(1, len(backprop_central_tendency_accuracies_doubled_in_percent) + 1),
            backprop_central_tendency_accuracies_doubled_in_percent,
            label='Backpropagation Median' if backprop_median else 'Backpropagation Mean',
            color='red', linestyle='--', linewidth=5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    ax.text(1, max(backprop_central_tendency_accuracies_doubled_in_percent) + 1,
            'Backpropagation Median' if backprop_median else 'Backpropagation Mean', fontsize=50, color='red')

    boxprops = dict(linestyle='-', linewidth=4)
    whiskerprops = dict(linestyle='-', linewidth=4)
    capprops = dict(linestyle='-', linewidth=4)
    flierprops = dict(marker='o', markersize=14, markeredgewidth=4,
                      linestyle='none')
    medianprops = dict(linestyle='-', linewidth=4, color='firebrick')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    meanlineprops = dict(linestyle='-.', linewidth=4, color='purple')

    bp = ax.boxplot(accuracies_in_percent, patch_artist=True, widths=0.6, flierprops=flierprops,
                    boxprops=boxprops, medianprops=medianprops, meanprops=meanlineprops,
                    whiskerprops=whiskerprops, capprops=capprops,
                    meanline=True, showmeans=False)

    for patch, color in zip(bp['boxes'], box_colors * 4):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(4)

    for median, color in zip(bp['medians'], 4*["cornflowerblue", "palevioletred"]):
        median.set_color(color)
        median.set_linewidth(4)

    ax.set_xticks([1.5, 3.5, 5.5, 7.5])
    ax.set_xticklabels(NN_architectures)
    ax.tick_params(width=5, length=10)

    ax.set_ylim(25, 110)
    y_ticks_to_hide = [9]

    # Hide tick lines
    for tic in y_ticks_to_hide:
        ax.get_yticklines()[tic * 2].set_visible(False)  # Hide the lower tick line
        ax.get_yticklines()[tic * 2 + 1].set_visible(False)  # Hide the upper tick line

    # Hide tick labels
    y_labels = ax.get_yticklabels()
    for i in y_ticks_to_hide:
        y_labels[i].set_visible(False)

    config1 = mpatches.Patch(color=box_colors[0])
    config2 = mpatches.Patch(color=box_colors[1])
    config_names = ["GA", "GA (static AB)" if dataset_name == "load_digits" else "GA (adaptive AB)"]
    plt.legend(labels=config_names, handles=[config1, config2], loc='lower right', fontsize=40, ncol = 2)

    plt.xlabel("Architectures", fontsize=50)
    plt.ylabel("Test Accuracy (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if grid: ax.grid(visible=True, which='both', linestyle='-', axis='y')
    plt.savefig(
        f"{results_path}/plots/GA_performance_comparison_boxplot_{dataset_name}"
        f"{'_training_accuracy' if training_accuracy else '_testing_accuracy'}{'_grid' if grid else ''}"
        f"{'_backprop_median' if backprop_median else '_backprop_mean'}.png")

def create_accuracy_box_plot_make_blobs_load_digits_fewer_classes(dataset_name, training_accuracy, data=None, grid=False, evaluate=None, source="tmp_log"):
    data = get_make_blobs_load_digits_fewer_classes_data(dataset_name, get_top_accuracy, training_accuracy, evaluate, source) if not data else data
    accuracies = list(data.values())
    accuracies_in_percent = []
    for acc in accuracies:
        acc_in_percentage = []
        for i in acc:
            acc_in_percentage.append(i * 100)
        accuracies_in_percent.append(acc_in_percentage)
    n_classes = range(2, 11) if dataset_name == "make_blobs" else [2, 3, 4, 5, 10]

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))

    ax = fig.add_subplot(111)
    ax.set_xticklabels(n_classes)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    boxprops = dict(linestyle='-', linewidth=4)
    whiskerprops = dict(linestyle='-', linewidth=4)
    capprops = dict(linestyle='-', linewidth=4)
    flierprops = dict(marker='o', markersize=14, markeredgewidth=4,
                      linestyle='none')
    medianprops = dict(linestyle='-', linewidth=4, color='firebrick')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    meanlineprops = dict(linestyle='-.', linewidth=4, color='purple')

    bp = ax.boxplot(accuracies_in_percent, patch_artist=True, widths=0.6, flierprops=flierprops,
                    boxprops=boxprops, medianprops=medianprops, meanprops=meanlineprops,
                    whiskerprops=whiskerprops, capprops=capprops,
                    meanline=True, showmeans=False)

    box_colors = ['green', 'blue', 'gold', 'violet', 'yellowgreen', 'orange', 'cyan', 'pink', 'darkblue', 'lime'] if dataset_name == "make_blobs" else ['green', 'blue', 'gold', 'violet', 'yellowgreen']
    median_colors = ['darkgreen', 'cornflowerblue', 'brown', 'purple', 'forestgreen', 'darkorange', 'deepskyblue', 'hotpink', 'royalblue', 'mediumseagreen'] if dataset_name == "make_blobs" else ['green', 'cornflowerblue', 'brown', 'red', 'green']

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(4)

    for median, color in zip(bp['medians'], median_colors):
        median.set_color(color)
        median.set_linewidth(4)

    ax.set_ylim(50, 105)

    """y_ticks_to_hide = [8 if dataset_name == "make_blobs" else 9]

    # Hide tick lines
    for tic in y_ticks_to_hide:
        ax.get_yticklines()[tic * 2].set_visible(False)  # Hide the lower tick line
        ax.get_yticklines()[tic * 2 + 1].set_visible(False)  # Hide the upper tick line

    # Hide tick labels
    y_labels = ax.get_yticklabels()
    for i in y_ticks_to_hide:
        y_labels[i].set_visible(False)"""

    plt.xlabel("No. Classes", fontsize=50)
    plt.ylabel("Test Accuracy (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if grid: ax.grid(visible=True, which='both', linestyle='-', axis='y')
    plt.savefig(f"{results_path}/plots/GA_performance_comparison_boxplot_{dataset_name}_{'training_accuracy' if training_accuracy else 'testing_accuracy'}{'_grid' if grid else ''}.png")


def create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes(dataset_name, training_accuracy, data=None, grid=False, evaluate=None, source="tmp_log"):
    data = get_make_blobs_load_digits_fewer_classes_data(dataset_name, get_top_accuracy, training_accuracy, evaluate, source) if not data else data
    accuracies = list(data.values())
    accuracies_in_percent = []
    for acc in accuracies:
        acc_in_percentage = []
        for i in acc:
            acc_in_percentage.append(i * 100)
        accuracies_in_percent.append(acc_in_percentage)
    dataset_instances = list(data.keys())
    n_classes = range(2, 11) if "make_blobs" in dataset_name else [2, 3, 4, 5, 10]

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))

    ax = fig.add_subplot(111)
    np.random.seed(43)
    colors = np.random.rand(9, 3)

    for i in range(len(dataset_instances)):
        ax.scatter([i+2] * len(accuracies_in_percent[i]), accuracies_in_percent[i], s=400, c=[colors[n_classes[i]-2]] * len(accuracies_in_percent[i]), alpha=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10]) if "make_blobs" in dataset_name else ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels(n_classes)
    ax.tick_params(width=5, length=10)

    ax.set_ylim(40, 110)

    y_ticks_to_hide = [7]

    # Hide tick lines
    for tic in y_ticks_to_hide:
        ax.get_yticklines()[tic * 2].set_visible(False)  # Hide the lower tick line
        ax.get_yticklines()[tic * 2 + 1].set_visible(False)  # Hide the upper tick line

    # Hide tick labels
    y_labels = ax.get_yticklabels()
    for i in y_ticks_to_hide:
        y_labels[i].set_visible(False)

    plt.xlabel("No. Classes", fontsize=50)
    plt.ylabel("Test Accuracy (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if grid: ax.grid(visible=True, which='both', linestyle='-', axis='y')
    plt.savefig(f"{results_path}/plots/GA_performance_comparison_scatterplot_{dataset_name}_{'training_accuracy' if training_accuracy else 'testing_accuracy'}{'_grid' if grid else ''}.png")

def get_sparsity_development_data(dataset_name):
    dataset_path = f"{results_path}/{dataset_name}/"
    if "load_digits_fewer_classes" in dataset_name:
        dataset_instances = ["load_digits_binary", "load_digits_ternary", "load_digits_quaternary",
                             "load_digits_quinary", "load_digits_denary"]
    else:
        dataset_instances = ["make_two_blobs", "make_three_blobs", "make_four_blobs", "make_five_blobs",
                             "make_six_blobs", "make_seven_blobs", "make_eight_blobs", "make_nine_blobs",
                             "make_ten_blobs"]
    test_runs = {dataset_instance: [name for name in os.listdir(f"{results_path}/{dataset_name}/{dataset_instance}/") if
                                    name[:4] == "job_"] for dataset_instance in dataset_instances}
    data = {dataset_instance: [] for dataset_instance in dataset_instances}
    print(data.keys())
    for dataset_instance in dataset_instances:
        for test_run in test_runs[dataset_instance]:
            with open(f"{dataset_path}/{dataset_instance}/{test_run}/pickled_objects/pickled_tmp_log", "rb") as file:
                tmp_log = dill.load(file)
            sparsity_development = tmp_log["max_pruned_SLT"]["sparsity_development"].split('->')
            initial_sparsity = float(sparsity_development[0].replace('%', '').strip())
            final_sparsity = float(sparsity_development[1].replace('%', '').strip())
            data[dataset_instance].append((initial_sparsity, final_sparsity))

    return data


def plot_sparsity_increase_bar_plot(dataset_name, grid=False):
    data = get_sparsity_development_data(dataset_name)

    mean_initial_sparsities = {dataset_instance: 0 for dataset_instance in data.keys()}
    mean_final_sparsities = {dataset_instance: 0 for dataset_instance in data.keys()}

    for dataset_instance in data.keys():
        initial_sparsities = [initial_sparsity for initial_sparsity, _ in data[dataset_instance]]
        final_sparsities = [final_sparsity for _, final_sparsity in data[dataset_instance]]
        mean_initial_sparsities[dataset_instance] = np.mean(initial_sparsities)
        mean_final_sparsities[dataset_instance] = np.mean(final_sparsities)

    bar_width = 0.35  # Bar width
    num_classes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]) if "make_blobs" in dataset_name else np.array([2, 3, 4, 5, 10])
    x_labels = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]) if "make_blobs" in dataset_name else np.array([2, 3, 4, 5, 6])

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig, ax = plt.subplots(figsize=(18, 13))

    bars1 = ax.bar(x_labels, list(mean_initial_sparsities.values()), bar_width, label='Sparsity After Evolution',
                   color='cornflowerblue', edgecolor='black', linewidth=1.2, alpha=0.7)
    bars2 = ax.bar(np.array(x_labels) + bar_width, list(mean_final_sparsities.values()), bar_width, label='Sparsity After Post-Evolutionary Pruning',
                   color='palevioletred', edgecolor='black', linewidth=1.2, alpha=0.7)

    # Add mean values on top of bars (rounded to 1 decimal place)
    for i, bar in enumerate(bars1):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', fontsize=35, color='black')

    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}', ha='center', fontsize=35, color='black')

    ax.set_xticks([2.18, 3.18, 4.18, 5.18, 6.18, 7.18, 8.18, 9.18, 10.18]) if "make_blobs" in dataset_name else ax.set_xticks([2.18, 3.18, 4.18, 5.18, 6.18])
    ax.set_xticklabels(num_classes)
    ax.tick_params(width=5, length=10)

    plt.ylim(0, 120, 10)
    ax.set_yticks(ax.get_yticks()[:-1])  # Remove the last y-tick

    plt.xlabel("No. Classes", fontsize=50)
    plt.ylabel("Mean Sparsity (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if grid: ax.grid(visible=True, which='both', linestyle='-', axis='y')

    ax.legend(fontsize=35, loc='upper right', edgecolor='black', ncol=1)

    plt.tight_layout()
    plt.savefig(f"{results_path}/plots/sparsity_development_{dataset_name}.pdf")


if __name__ == "__main__":

    make_moons_box_colors = make_circles_box_colors = load_digits_binary_box_colors = ["lightblue", "pink"]
    load_digits_box_colors = ["lightblue", "purple"]
    # Create GA_performance_comparison_boxplot_make_moons...
    create_accuracy_box_plot_make_moons_make_circles_load_digits_both_configs("make_moons", make_moons_box_colors, False, False, backprop_median=False)
    # Create GA_performance_comparison_boxplot_make_circles...
    create_accuracy_box_plot_make_moons_make_circles_load_digits_both_configs("make_circles", make_circles_box_colors,False, False, backprop_median=False)
    # Create GA_performance_comparison_boxplot_load_digits...
    create_accuracy_box_plot_make_moons_make_circles_load_digits_both_configs("load_digits", load_digits_box_colors,False)
    # Create GA_performance_comparison_boxplot_load_digits...
    create_accuracy_box_plot_make_moons_make_circles_load_digits_both_configs("load_digits_binary", load_digits_binary_box_colors,False)
    # Create GA_performance_comparison_scatterplot_make_blobs...
    create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes("make_blobs", False, grid=True)
    # Create GA_performance_comparison_scatterplot_load_digits_fewer_classes...
    create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes("load_digits_fewer_classes", False, grid=True)

    # GA_performance_comparison_boxplot_make_blobs...
    create_accuracy_box_plot_make_blobs_load_digits_fewer_classes("make_blobs", False)
    # GA_performance_comparison_boxplot_load_digits_fewer_classes..."""
    create_accuracy_box_plot_make_blobs_load_digits_fewer_classes("load_digits_fewer_classes", False)
    evaluate = {"model_args": {"nn_architecture":[64, 75], "init":"uniform", "prune_rate":0.5, "activation":"relu", "mask_except":["net.0.bias", "net.2.bias"], "device":"cpu"}, "dataset_args":{"name": "digits", "norm": False, "device": "cpu"}}
    create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes("load_digits_fewer_classes", False, grid=True, evaluate=evaluate)

    evaluate = {"model_args": {"nn_architecture": [64, 75], "init": "uniform", "prune_rate": 0.5, "activation": "relu",
                               "mask_except": ["net.0.bias", "net.2.bias"], "device": "cpu"},
                "dataset_args": {"name": "digits", "norm": True, "device": "cpu"}}
    data = get_make_blobs_load_digits_fewer_classes_data_backprop("load_digits_fewer_classes")
    calculate_statistics_for_load_digits_fewer_classes_and_make_blobs_dataset("load_digits_fewer_classes", False, training_accuracy=False, evaluate=evaluate, source="tmp_log", data=data)
    create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes("load_digits_fewer_classes", False, grid=True, evaluate=evaluate, source="tmp_log")
    create_accuracy_box_plot_make_blobs_load_digits_fewer_classes("load_digits_fewer_classes", False, grid=True, evaluate=evaluate, source="tmp_log")

    # print(np.median(get_make_moons_make_circles_load_digits_data("make_moons", config_name[1], get_top_accuracy, False)["[2, 20, 2]"])) -> 0.92065625
    # print(np.median(get_accuracy_data("make_moons")["[2, 20, 2]"])) -> 0.999875

    # print(np.max(get_make_moons_make_circles_load_digits_data("load_digits", config_name[2], get_top_accuracy, False)["[64, 75, 10]"]))

    # Gather data for Figure 3 Table
    # get_and_store_make_moons_make_circles_mean_median_std(config_number=0, metric_method_name="get_top_accuracy")
    # get_and_store_make_moons_make_circles_mean_median_std(config_number=1, metric_method_name="get_top_accuracy_individuals_sparsity")

    plot_sparsity_increase_bar_plot("load_digits_fewer_classes")