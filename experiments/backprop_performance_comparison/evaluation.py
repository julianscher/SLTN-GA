import dill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.ga_performance_comparison.statistics import get_mean_confidence_interval_short
from utilities.helper_functions import get_results_path
from scipy.stats import t

# Experiment contents overview
NN_architectures_binary = [[2, 20, 2], [2, 75, 2], [2, 100, 2], [2, 50, 50, 2]]
NN_architectures_digits = [[64, 20, 10], [64, 75, 10], [64, 100, 10], [64, 50, 50, 10]]
NN_architectures_digits_binary = [[64, 20, 2], [64, 75, 2], [64, 100, 2], [64, 50, 50, 2]]

results_path = f"{get_results_path()}/backprop_performance_results"

def get_NN_architectures(dataset_name):
    if dataset_name in ["make_moons", "make_circles"]:
        return NN_architectures_binary
    elif dataset_name == "load_digits":
        return NN_architectures_digits
    else:
        return NN_architectures_digits_binary


def get_data(dataset_name):
    NN_architectures = get_NN_architectures(dataset_name)
    data = {str(NN_architecture): None for NN_architecture in NN_architectures}
    for NN_architecture in NN_architectures:
        with open(f"{results_path}/{dataset_name}/pickled_backprop_data_{NN_architecture}_{dataset_name}", 'rb') as pickle_file:
            data[str(NN_architecture)] = dill.load(pickle_file)

    return data


def get_make_blobs_load_digits_fewer_classes_data_backprop(dataset_name):
    dataset_path = f"{results_path}/{dataset_name}/"
    if dataset_name == "load_digits_fewer_classes":
        dataset_instances = ["load_digits_binary", "load_digits_ternary", "load_digits_quaternary",
                             "load_digits_quinary", "load_digits_denary"]
        NN_architectures = [[64, 75, 2], [64, 75, 3], [64, 75, 4], [64, 75, 5], [64, 75, 10]]
    else:
        dataset_instances = ["make_two_blobs", "make_three_blobs", "make_four_blobs", "make_five_blobs",
                             "make_six_blobs", "make_seven_blobs", "make_eight_blobs", "make_nine_blobs",
                             "make_ten_blobs"]
        NN_architectures = [[2, 100, 2], [2, 100, 3], [2, 100, 4], [2, 100, 5], [2, 100, 6], [2, 100, 7], [2, 100, 8], [2, 100, 9], [2, 100, 10]]
    data = {dataset_instance: [] for dataset_instance in dataset_instances}
    for dataset_instance, NN_architecture in zip(dataset_instances, NN_architectures):
        with open(f"{dataset_path}/{dataset_instance}/pickled_backprop_data_{NN_architecture}_{dataset_instance}", 'rb') as pickle_file:
            data[dataset_instance] = dill.load(pickle_file)['final_accuracies']

    return data


def get_accuracy_data(dataset_name):
    data = get_data(dataset_name)
    NN_architectures = get_NN_architectures(dataset_name)
    accuracy_data = {str(NN_architecture): [] for NN_architecture in NN_architectures}
    for key in data.keys():
        accuracy_data[key] = data[key].get("final_accuracies")
    return accuracy_data

def get_training_time_data(dataset_name):
    data = get_data(dataset_name)
    NN_architectures = get_NN_architectures(dataset_name)
    accuracy_data = {str(NN_architecture): [] for NN_architecture in NN_architectures}
    for key in data.keys():
        accuracy_data[key] = data[key].get("training_times")
    return accuracy_data

def get_mean_accuracy_for_architecture(NN_architecture, dataset_name):
    backprop_data = get_accuracy_data(dataset_name)
    return np.mean(backprop_data.get(str(NN_architecture)))


def get_median_accuracy_for_architecture(NN_architecture, dataset_name):
    backprop_data = get_accuracy_data(dataset_name)
    return np.median(backprop_data.get(str(NN_architecture)))


def get_std_accuracy_for_architecture(NN_architecture, dataset_name):
    backprop_data = get_accuracy_data(dataset_name)
    return np.std(backprop_data.get(str(NN_architecture)))


def get_95_percent_confidence_interval(NN_architecture, dataset_name):
    backprop_data = np.array(get_accuracy_data(dataset_name).get(str(NN_architecture)))
    # Calculate standard error
    SEM = np.std(backprop_data, axis=0, ddof=1) / np.sqrt(backprop_data.shape[0])

    # Calculate the 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = np.array(backprop_data).shape[0] - 1
    confidence_interval = SEM * t.ppf((1 + confidence_level) / 2.0, degrees_freedom)
    print(get_mean_confidence_interval_short(backprop_data))
    print(f"Confidence interval: {confidence_interval}")
    return confidence_interval


def get_and_store_make_moons_make_circles_mean_median_std():
    make_moons_data = get_accuracy_data("make_moons")
    make_moons_data_2_20_2 = make_moons_data["[2, 20, 2]"]
    make_moons_data_2_75_2 = make_moons_data["[2, 75, 2]"]
    make_moons_data_2_100_2 = make_moons_data["[2, 100, 2]"]
    make_moons_data_2_50_50_2 = make_moons_data["[2, 50, 50, 2]"]

    make_circles_data = get_accuracy_data("make_circles")
    make_circles_data_2_20_2 = make_circles_data["[2, 20, 2]"]
    make_circles_data_2_75_2 = make_circles_data["[2, 75, 2]"]
    make_circles_data_2_100_2 = make_circles_data["[2, 100, 2]"]
    make_circles_data_2_50_50_2 = make_circles_data["[2, 50, 50, 2]"]

    # Calculate statistics
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

    # Create DataFrame
    index_labels = ["2_20_2", "2_75_2", "2_100_2", "2_50_50_2"]
    df = pd.DataFrame(data, index=index_labels)

    # Save to Excel
    excel_path = f"{results_path}/statistics/moons_circles_mean_median_std_for_backpropagation.xlsx"
    df.to_excel(excel_path)

    print(f"Results saved to {excel_path}")


def get_mean_training_time_for_architecture(NN_architecture, dataset_name):
    backprop_data = get_training_time_data(dataset_name)
    return np.mean(backprop_data.get(str(NN_architecture)))


def get_median_training_time_for_architecture(NN_architecture, dataset_name):
    backprop_data = get_training_time_data(dataset_name)
    return np.median(backprop_data.get(str(NN_architecture)))


def create_accuracy_box_plot(dataset_name):
    data = list(get_accuracy_data(dataset_name).values())
    fig = plt.figure(figsize=(10, 10))

    # Creating axes instance
    ax = fig.add_subplot(111)

    ax.set_xticklabels(list(get_accuracy_data(dataset_name).keys()), rotation=90)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)
    bp = ax.boxplot(data, patch_artist=True)

    plt.ylabel("accuracy", fontsize=16)
    plt.title(f"Backpropagation_performance_comparison boxplot {dataset_name}", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{results_path}/plots/Backpropagation_performance_comparison_boxplot_{dataset_name}.png")


if __name__ == "__main__":
    # create_accuracy_box_plot("make_circles")
    # print(get_mean_accuracy_for_architecture("[2, 50, 50, 2]", "make_circles"))
    #get_and_store_make_moons_make_circles_mean_median_std()
    print(get_make_blobs_load_digits_fewer_classes_data_backprop("load_digits_fewer_classes"))
