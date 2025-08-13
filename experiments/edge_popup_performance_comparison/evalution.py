import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D

from experiments.backprop_performance_comparison.evaluation import get_accuracy_data, \
    get_mean_accuracy_for_architecture, \
    get_make_blobs_load_digits_fewer_classes_data_backprop
from experiments.ga_performance_comparison.evaluation import get_make_moons_make_circles_load_digits_data, \
    get_top_accuracy, create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes, \
    create_accuracy_box_plot_make_blobs_load_digits_fewer_classes, \
    calculate_statistics_for_load_digits_fewer_classes_and_make_blobs_dataset, \
    get_make_blobs_load_digits_fewer_classes_data
from experiments.ga_performance_comparison.statistics import get_mean_confidence_interval_short
from utilities.helper_functions import get_results_path

results_path = f"{get_results_path()}/edge_popup_performance_results"

# Experiment contents overview
configurations = ["fc_kn_unsigned", "fc_sc_unsigned", "fc_u_unsigned"]
NN_architectures_binary = ["2_20_2", "2_75_2", "2_100_2", "2_50_50_2"]
NN_architectures_digits = ["64_20_10", "64_75_10", "64_100_10", "64_50_50_10"]
NN_architectures_digits_binary = ["64_20_2", "64_75_2", "64_100_2", "64_50_50_2"]
dataset_names = ["make_moons", "make_circles", "load_digits", "load_digits_binary"]


def get_NN_architectures(dataset_name):
    if dataset_name in ["make_moons", "make_circles"]:
        return NN_architectures_binary
    elif dataset_name == "load_digits":
        return NN_architectures_digits
    else:
        return NN_architectures_digits_binary

def open_results_csv(path_to_csv):
    data = pd.read_csv(path_to_csv)
    return data


def get_config_results(config_name, path_to_csv):
    data = open_results_csv(path_to_csv)
    config_rows = []
    for row in data.itertuples():
        if config_name in str(row[2]):
            config_rows.append(row)
    return config_rows


def get_row_values(config_results, row_idx):
    values = []
    for row in config_results:
        values.append(row[row_idx])
    return values

def get_current_val_top1(config_results, NN_architecture):
    current_val_top_1 = []
    for row in config_results:
        if row[3][1:] == NN_architecture:
            current_val_top_1.append(row[5] / 100)# TODO
    return current_val_top_1

def get_best_val_top_1(config_results, NN_architecture):
    best_val_top_1 = []
    for row in config_results:
        if row[3][1:] == NN_architecture:
            best_val_top_1.append(row[7] / 100)
    return best_val_top_1


def get_best_train_top_1(config_results, NN_architecture):
    best_train_top_1 = []
    for row in config_results:
        if row[3][1:] == NN_architecture:
            best_train_top_1.append(row[9])
    return best_train_top_1


def get_data(dataset_name, configurations=configurations, path=results_path):
    NN_architectures = get_NN_architectures(dataset_name)
    data = {f"{configuration}_{NN_architecture}": []
            for configuration in configurations for NN_architecture in NN_architectures}

    results_csv_path = f"{path}/{dataset_name}/results.csv"
    for config in configurations:
        for NN_architecture in NN_architectures:
            config_results = get_config_results(config, results_csv_path)
            values = get_current_val_top1(config_results, NN_architecture)
            if len(values) > 25:
                values = values[:25]
            data[f"{config}_{NN_architecture}"].extend(values)

    for key in data.keys():
        print(f"{key} valid runs:", len(data[key]))
    return data


def get_load_digits_fewer_classes_make_blobs_data_fc_u_unsigned(dataset_name, path=results_path):
    if dataset_name == "load_digits_fewer_classes":
        dataset_instances = ["load_digits_binary", "load_digits_ternary", "load_digits_quaternary", "load_digits_quinary", "load_digits_denary"]
        NN_architectures = ["64_75_2", "64_75_3", "64_75_4", "64_75_5", "64_75_10"]
    elif dataset_name == "make_blobs":
        dataset_instances = ["make_two_blobs", "make_three_blobs", "make_four_blobs", "make_five_blobs", "make_six_blobs", "make_seven_blobs", "make_eight_blobs", "make_nine_blobs", "make_ten_blobs"]
        NN_architectures = ["2_100_2", "2_100_3", "2_100_4", "2_100_5", "2_100_6", "2_100_7", "2_100_8", "2_100_9", "2_100_10"]

    data = {dataset_instance: [] for dataset_instance in dataset_instances}
    config = "fc_u_unsigned"

    for dataset_instance in dataset_instances:
        for NN_architecture in NN_architectures:
            results_csv_path = f"{path}/{dataset_name}/{dataset_instance}/results.csv"
            config_results = get_config_results(config, results_csv_path)
            values = get_current_val_top1(config_results, NN_architecture)
            if len(values) > 25:
                values = values[:25]
            data[dataset_instance].extend(values)

    for key in data.keys():
        print(f"{key} valid runs:", len(data[key]))
    return data


def get_mean_accuracies(dataset_name):
    data = get_data(dataset_name)
    mean_accuracies = {config: 0 for config in data.keys()}
    for key in data.keys():
        mean_accuracies[key] = np.mean(data[key])
    return mean_accuracies


def get_median_accuracies(dataset_name):
    data = get_data(dataset_name)
    median_accuracies = {config: 0 for config in data.keys()}
    for key in data.keys():
        median_accuracies[key] = np.median(data[key])
    return median_accuracies

def get_std_accuracies(dataset_name):
    data = get_data(dataset_name)
    std_accuracies = {config: 0 for config in data.keys()}
    for key in data.keys():
        std_accuracies[key] = np.std(data[key])
    return std_accuracies


def mystep(x, y, ax=None, where='post', **kwargs):
    """ Display only horizontal lines of plot
    cf. https://stackoverflow.com/questions/44961184/matplotlib-plot-only-horizontal-lines-in-step-plot"""
    assert where in ['post', 'pre']
    x = np.array(x)
    y = np.array(y)
    if where == 'post': y_slice = y[:-1]
    if where == 'pre': y_slice = y[1:]
    print(y_slice)
    X = np.c_[x[:-1], x[1:], x[1:]]
    Y = np.c_[y_slice, y_slice, np.zeros_like(x[:-1]) * np.nan]
    return X.flatten(), Y.flatten()


def create_accuracy_box_plot(dataset_name, data, x_labels, grid=False):
    data_in_percent = []
    for acc_list in data:
        data_in_percent.append([100*acc for acc in acc_list])

    backprop_accuracies = get_accuracy_data(dataset_name)
    backprop_mean_accuracies_in_percent = [100 * get_mean_accuracy_for_architecture(NN_architecture, dataset_name) for
                                           NN_architecture in backprop_accuracies.keys()]
    backprop_mean_accuracies_in_percent_repeated = []
    for mean_acc in backprop_mean_accuracies_in_percent:
        backprop_mean_accuracies_in_percent_repeated.extend(3*[mean_acc])

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))

    ax = fig.add_subplot(111)
    ax.set_xticklabels(x_labels)

    """for i in range(0, len(backprop_mean_accuracies_in_percent_repeated), 3):
        x1, y1 = i+1, backprop_mean_accuracies_in_percent_repeated[i]
        x2, y2 = i+3, backprop_mean_accuracies_in_percent_repeated[i+2]
        ax.hlines(y=y1, xmin=x1, xmax=x2, colors='red', linestyles="--", linewidth=5)"""
    ax.step(range(1, len(backprop_mean_accuracies_in_percent_repeated)+1), backprop_mean_accuracies_in_percent_repeated,
            color='red', linestyle='--', linewidth=5)

    ax.text(1, max(backprop_mean_accuracies_in_percent)+1, "Backpropagation Mean", fontsize=50, color='red')

    """ax.scatter(range(1, len(backprop_mean_accuracies_in_percent_repeated) + 1), backprop_mean_accuracies_in_percent_repeated, color='red',
               label='Backpropagation Mean', marker='o', linewidths=3, s=300)"""
    # ax.plot(range(1, len(backprop_median_accuracies) + 1), backprop_median_accuracies, color='red', label='Backpropagation Median', marker='o', linestyle='--')
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

    bp = ax.boxplot(data_in_percent, patch_artist=True, widths=0.6, flierprops=flierprops,
                    boxprops=boxprops, medianprops=medianprops, meanprops=meanlineprops,
                    whiskerprops=whiskerprops, capprops=capprops,
                    meanline=True, showmeans=False)
    for patch, color in zip(bp['boxes'], 4*["gold", "lightgrey", "cornflowerblue"]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_linewidth(4)

    for median, color in zip(bp['medians'], 4*["orange", "grey", "midnightblue"]):
        median.set_color(color)
        median.set_linewidth(4)

    ax.set_ylim(25, 110)
    ax.tick_params(width=5, length=10)


    x_ticks_to_hide = [0, 2, 3, 5, 6, 8, 9, 11]
    y_ticks_to_hide = [9]

    # Hide tick lines
    for tic in x_ticks_to_hide:
        ax.get_xticklines()[tic * 2].set_visible(False)  # Hide the lower tick line
        ax.get_xticklines()[tic * 2 + 1].set_visible(False)  # Hide the upper tick line
    for tic in y_ticks_to_hide:
        ax.get_yticklines()[tic * 2].set_visible(False)  # Hide the lower tick line
        ax.get_yticklines()[tic * 2 + 1].set_visible(False)  # Hide the upper tick line

    # Hide tick labels
    x_labels = ax.get_xticklabels()
    for i in x_ticks_to_hide:
        x_labels[i].set_visible(False)
    y_labels = ax.get_yticklabels()
    for i in y_ticks_to_hide:
        y_labels[i].set_visible(False)


    fc_kn_unsigned = mpatches.Patch(color="gold")
    fc_sc_unsigned = mpatches.Patch(color="lightgrey")
    fc_u_unsigned = mpatches.Patch(color="cornflowerblue")
    config_names = ["W $\\sim \\mathcal{N}_{k}$", "W $\\sim U_{k}$", "W $\\sim \\mathcal{U}_{[-1, 1]}$"]
    plt.legend(labels=config_names, handles=[fc_kn_unsigned, fc_sc_unsigned, fc_u_unsigned], loc='lower right', fontsize=40,
               mode = "expand", ncol = 3)

    plt.xlabel("Architectures", fontsize=50)
    plt.ylabel("Test Accuracy (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if grid: ax.grid(visible=True, which='both', linestyle='-', axis='y')
    plt.savefig(f"{results_path}/plots/Edge_popup_performance_comparison_boxplot_{dataset_name}{'_grid' if grid else ''}.png")


def convert_to_percentages(data):
    percentage_data = {}
    for key, values in data.items():
        percentage_values = [value * 100 for value in values]
        percentage_data[key] = percentage_values
    return percentage_data


def create_accuracy_box_plot_varied_prune_rates(dataset_name, considered_GA_config, x_labels, file_name_extension, GA_accuracies=None, grid=False):
    # Get backpropagation data
    backprop_accuracies = get_accuracy_data(dataset_name)
    backprop_accuracies_in_percent = convert_to_percentages(backprop_accuracies)
    backprop_mean_accuracies_in_percent = [100*get_mean_accuracy_for_architecture(NN_architecture, dataset_name) for
                                           NN_architecture in backprop_accuracies.keys()]

    backprop_accuracies_confidence_interval = np.array(
        [get_mean_confidence_interval_short(backprop_accuracies_in_percent.get(NN_architecture)) for
         NN_architecture in backprop_accuracies_in_percent.keys()])

    lower_bound_backprop = backprop_accuracies_confidence_interval[:, 0]
    upper_bound_backprop = backprop_accuracies_confidence_interval[:, 1]
    print(f"lower_bound_backprop: {lower_bound_backprop}")
    print(f"upper_bound_backprop: {upper_bound_backprop}")

    # Get GA data for both configurations
    if not GA_accuracies:
        GA_data_dict = {"top_speed": get_make_moons_make_circles_load_digits_data(dataset_name, "GA_evolutions_top_speed", get_top_accuracy, False),
                        "max_acc_bound": get_make_moons_make_circles_load_digits_data(dataset_name, "GA_evolutions_max_acc_bound", get_top_accuracy, False)}
        GA_accuracies = {NN_architecture: GA_data_dict[config][NN_architecture] for (config, NN_architecture)
                         in zip(considered_GA_config, backprop_accuracies.keys())}
    GA_accuracies_in_percent = convert_to_percentages(GA_accuracies)
    GA_mean_accuracies_in_percent = [np.mean(GA_accuracies_in_percent.get(NN_architecture)) for NN_architecture in GA_accuracies_in_percent.keys()]
    GA_accuracies_confidence_interval = np.array([get_mean_confidence_interval_short(GA_accuracies_in_percent.get(NN_architecture))
                                                  for NN_architecture in GA_accuracies_in_percent.keys()])


    lower_bound_GA = GA_accuracies_confidence_interval[:, 0]
    upper_bound_GA = GA_accuracies_confidence_interval[:, 1]
    print(f"lower_bound_GA: {lower_bound_GA}")
    print(f"upper_bound_GA: {upper_bound_GA}")

    # Get edge-popup data
    edge_popup_accuracies = get_data(dataset_name, ["fc_u_unsigned"], f"{results_path}/make_moons_make_circles_varied_pruning_rate")
    edge_popup_accuracies_in_percent = convert_to_percentages(edge_popup_accuracies)
    edge_popup_mean_accuracies_in_percent = [np.mean(edge_popup_accuracies_in_percent.get(NN_architecture))
                                  for NN_architecture in edge_popup_accuracies_in_percent.keys()]
    edge_popup_accuracies_confidence_interval = np.array(
        [get_mean_confidence_interval_short(edge_popup_accuracies_in_percent.get(NN_architecture)) for
         NN_architecture in edge_popup_accuracies_in_percent.keys()])


    lower_bound_edge_popup = edge_popup_accuracies_confidence_interval[:, 0]
    upper_bound_edge_popup = edge_popup_accuracies_confidence_interval[:, 1]
    print(f"lower_bound_edge_popup: {lower_bound_edge_popup}")
    print(f"upper_bound_edge_popup: {upper_bound_edge_popup}")

    # Get edge-popup data with 50% pruned
    edge_popup_original_accuracies = get_data(dataset_name, ["fc_u_unsigned"], results_path)
    edge_popup_original_accuracies_in_percent = convert_to_percentages(edge_popup_original_accuracies)
    edge_popup_original_mean_accuracies_in_percent = [np.mean(edge_popup_original_accuracies_in_percent.get(NN_architecture))
                                                      for NN_architecture in edge_popup_original_accuracies_in_percent.keys()]
    edge_popup_original_accuracies_confidence_interval = np.array(
        [get_mean_confidence_interval_short(edge_popup_original_accuracies_in_percent.get(NN_architecture)) for
         NN_architecture in edge_popup_original_accuracies_in_percent.keys()])


    lower_bound_edge_popup_original = edge_popup_original_accuracies_confidence_interval[:, 0]
    upper_bound_edge_popup_original = edge_popup_original_accuracies_confidence_interval[:, 1]
    print(f"lower_bound_edge_popup_original: {lower_bound_edge_popup_original}")
    print(f"upper_bound_edge_popup_original: {upper_bound_edge_popup_original}")

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))


    ax = fig.add_subplot(111)
    x_tick_positions = [1.5, 3.5, 5.5, 7.5]

    ax.text(1.5, max(backprop_mean_accuracies_in_percent) + 1, "Backpropagation", fontsize=50, color='red')


    ax.plot(x_tick_positions, backprop_mean_accuracies_in_percent, color='red',
            label='Backpropagation', linestyle="--", linewidth=4)
    ax.plot(x_tick_positions, edge_popup_original_mean_accuracies_in_percent,
            color='blue', label='Edge-popup (50% Sparsity)', marker='d', markersize=20, linestyle="--", linewidth=4)
    ax.plot(x_tick_positions, edge_popup_mean_accuracies_in_percent, color='green',
            label='Edge-popup', marker='d', markersize=20, linestyle="-", linewidth=4)
    plt.plot(x_tick_positions, GA_mean_accuracies_in_percent, color='black', markersize=20, linestyle="--", linewidth=4)

    for idx, config in enumerate(considered_GA_config):
        plt.plot(x_tick_positions[idx], GA_mean_accuracies_in_percent[idx], marker='o', markersize=20,
                 color="orange" if config == "top_speed" else "violet", linewidth=4)


    ax.fill_between(x_tick_positions, lower_bound_backprop, upper_bound_backprop,
                    color='red', alpha=0.3)
    ax.fill_between(x_tick_positions, lower_bound_edge_popup_original,
                    upper_bound_edge_popup_original, color='blue', alpha=0.3)
    ax.fill_between(x_tick_positions, lower_bound_edge_popup, upper_bound_edge_popup,
                    color='green', alpha=0.3)
    ax.fill_between(x_tick_positions, lower_bound_GA, upper_bound_GA, color='black', alpha=0.3)

    legend_elements = [Line2D([0], [0],  marker='o', color='w', label='GA (adaptive AB)',
                              markerfacecolor='violet', markersize=20),
                       Line2D([0], [0], marker='o', color='w', label='GA',
                              markerfacecolor='orange', markersize=20),
                       Line2D([0], [0], marker='d', color='w', label='EP (50%)',
                              markerfacecolor='blue', markersize=20),
                       Line2D([0], [0], marker='d', color='w', label='EP (adapted)',
                              markerfacecolor='green', markersize=20)]


    ax.set_xticks([1.5, 3.5, 5.5, 7.5])
    ax.set_xticklabels(x_labels)


    ax.set_ylim(25, 110)

    # Hide tick lines
    y_ticks_to_hide = [9]
    for tic in y_ticks_to_hide:
        ax.get_yticklines()[tic * 2].set_visible(False)  # Hide the lower tick line
        ax.get_yticklines()[tic * 2 + 1].set_visible(False)  # Hide the upper tick line

    # Hide tick labels
    y_labels = ax.get_yticklabels()
    for i in y_ticks_to_hide:
        y_labels[i].set_visible(False)

    ax.tick_params(width=5, length=10)

    plt.legend(handles=legend_elements, loc='lower right', fontsize=40, mode = "expand", ncol = 2)
    plt.xlabel("Architectures + Sparsities", fontsize=50)
    plt.ylabel("Mean Accuracy (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    if grid: ax.grid(visible=True, which='both', linestyle='-', axis='y')
    plt.savefig(
        f"{results_path}/plots/Edge_popup_performance_comparison_boxplot_{dataset_name}_{file_name_extension}{'_grid' if grid else ''}.png")


def sanity_check(dataset_name):
    dataset_path = f"{results_path}/{dataset_name}"
    configs = [name for name in os.listdir(dataset_path) if os.path.isdir(f"{dataset_path}/{name}")]
    for config in configs:
        architectures = [name for name in os.listdir(f"{dataset_path}/{config}") if os.path.isdir(f"{dataset_path}/{config}/{name}")]
        for architecture in architectures:
            prune_rates = [name for name in os.listdir(f"{dataset_path}/{config}/{architecture}") if os.path.isdir(f"{dataset_path}/{config}/{architecture}/{name}")]
            for prune_rate in prune_rates:
                runs = [name for name in os.listdir(f"{dataset_path}/{config}/{architecture}/{prune_rate}") if os.path.isdir(f"{dataset_path}/{config}/{architecture}/{prune_rate}/{name}")]
                for run in runs:
                    with open(f"{dataset_path}/{config}/{architecture}/{prune_rate}/{run}/settings.txt") as txt_file:
                        print(txt_file.read())


def create_make_blobs_load_digits_fewer_classes_combined_error_plot(dataset_name, norm=True, shifted=True,
                                                                    grid=True, with_additional_GA=False):
    num_classes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]) if "make_blobs" in dataset_name else np.array(
        [2, 3, 4, 5, 10])
    x_labels = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]) if "make_blobs" in dataset_name else np.array([2, 3, 4, 5, 6])

    np.random.seed(42)

    # get_make_blobs_load_digits_fewer_classes_data_backprop("make_blobs")
    evaluate = {
        "model_args": {"nn_architecture": [2, 100] if "make_blobs" in dataset_name else [64, 75], "init": "uniform",
                       "prune_rate": 0.5, "activation": "relu",
                       "mask_except": ["net.0.bias", "net.2.bias"], "device": "cpu"},
        "dataset_args": {"name": "blobs" if "make_blobs" in dataset_name else "digits", "norm": norm,
                         "device": "cpu"}}
    accuracy_A_runs = np.array(
        list(get_make_blobs_load_digits_fewer_classes_data_backprop(dataset_name).values())).reshape(
        (len(num_classes), 25))
    accuracy_B_runs = np.array(list(
        get_make_blobs_load_digits_fewer_classes_data(dataset_name, get_top_accuracy, False, evaluate,
                                                      "tmp_log").values())).reshape((len(num_classes), 25))
    accuracy_C_runs = np.array(list(
        get_load_digits_fewer_classes_make_blobs_data_fc_u_unsigned(dataset_name, results_path).values())).reshape(
        (len(num_classes), 25))
    if with_additional_GA:
        if "make_blobs" in dataset_name:
            evaluate["dataset_args"]["norm"] = True
            accuracy_D_runs = np.array(list(
                get_make_blobs_load_digits_fewer_classes_data("make_blobs_normalized", get_top_accuracy, False,
                                                              evaluate, "tmp_log").values())).reshape(
                (len(num_classes), 25))
        else:
            evaluate["dataset_args"]["norm"] = False
            accuracy_D_runs = np.array(list(
                get_make_blobs_load_digits_fewer_classes_data("load_digits_fewer_classes_not_normalized",
                                                              get_top_accuracy, False, evaluate,
                                                              "tmp_log").values())).reshape((len(num_classes), 25))

    mean_A, std_A = accuracy_A_runs.mean(axis=1) * 100, accuracy_A_runs.std(axis=1) * 100
    mean_B, std_B = accuracy_B_runs.mean(axis=1) * 100, accuracy_B_runs.std(axis=1) * 100
    mean_C, std_C = accuracy_C_runs.mean(axis=1) * 100, accuracy_C_runs.std(axis=1) * 100
    if with_additional_GA:
        mean_D, std_D = accuracy_D_runs.mean(axis=1) * 100, accuracy_D_runs.std(axis=1) * 100

    # Cap the upper bound of error bars at 100%
    upper_A = np.minimum(mean_A + std_A, 100)
    upper_B = np.minimum(mean_B + std_B, 100)
    upper_C = np.minimum(mean_C + std_C, 100)
    if with_additional_GA:
        upper_D = np.minimum(mean_D + std_D, 100)

    # Lower bound remains the same
    lower_A = mean_A - std_A
    lower_B = mean_B - std_B
    lower_C = mean_C - std_C
    if with_additional_GA:
        lower_D = mean_D - std_D

    # Calculate asymmetric error bars
    err_A = [mean_A - lower_A, upper_A - mean_A]
    err_B = [mean_B - lower_B, upper_B - mean_B]
    err_C = [mean_C - lower_C, upper_C - mean_C]
    if with_additional_GA:
        err_D = [mean_D - lower_D, upper_D - mean_D]

    # Small horizontal shift for better separation
    if shifted:
        shift = 0.25 if "make_blobs" in dataset_name else 0.2
        num_classes_A = x_labels
        num_classes_B = x_labels
        num_classes_C = x_labels + shift
        if with_additional_GA:
            num_classes_D = x_labels + 1.5 * shift
    else:
        num_classes_A = x_labels
        num_classes_B = x_labels
        num_classes_C = x_labels
        num_classes_D = x_labels

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))
    ax = fig.add_subplot(111)

    plt.text(5.3 if "make_blobs" in dataset_name else 3.8, 101, "Backpropagation", color='red', fontsize=50,
             ha='right', va='bottom')


    if with_additional_GA:
        plt.errorbar(num_classes_D, mean_D, yerr=err_D, linestyle='-', elinewidth=5,
                     label="GA (normalized)" if "make_blobs" in dataset_name else "GA (not-normalized)",
                     marker="s", markersize=20, color='violet', ecolor='violet', linewidth=7, alpha=0.7)

    plt.errorbar(num_classes_A, mean_A, yerr=err_A, linestyle='--', elinewidth=5,
                 color='red', ecolor='red', linewidth=7)

    plt.errorbar(num_classes_B, mean_B, yerr=err_B, marker='o', linestyle='dotted', elinewidth=5,
                 label='GA', color='orange', ecolor='orange', markersize=30, linewidth=7)

    plt.errorbar(num_classes_C, mean_C, yerr=err_C, marker='d', linestyle='dotted', elinewidth=5,
                 label='EP (50%)', color='blue', ecolor='blue', markersize=30, linewidth=7)

    plt.xlabel("Number of Classes", fontsize=50)
    plt.ylabel("Accuracy (%)", fontsize=50)

    plt.xticks(x_labels, fontsize=50)
    plt.yticks(range(70, 110, 5), fontsize=50)
    if grid:
        plt.grid(axis='y', linestyle="--", alpha=0.7)

    ax = plt.gca()

    yticks = ax.get_yticks()
    xticks = ax.get_xticks()


    yticklabels = [str(tick) for tick in yticks]
    xticklabels = [str(tick) for tick in xticks]
    print(yticklabels)


    yticklabels[yticklabels.index('105')] = ''  # Change this for other y-ticks
    if dataset_name == "load_digits_fewer_classes":
        xticklabels[xticklabels.index('6')] = '10'

    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(width=5, length=10)

    ax.set_yticks(yticks[:-1])  # Remove the last y-tick entirely


    plt.legend(fontsize=40 if not with_additional_GA else 35, loc='lower left', edgecolor='black',
               ncol=2 if not with_additional_GA else 3)

    plt.savefig(
        f"{results_path}/ga_performance_results/plots/{dataset_name}_performance_comparison.pdf")

    plt.show()


if __name__ == '__main__':
    # Create Edge_popup_performance_comparison_boxplot_make_moons and
    moons_data = get_data("make_moons", ["fc_kn_unsigned", "fc_sc_unsigned", "fc_u_unsigned"], results_path)
    architecture_order = ['fc_kn_unsigned_2_20_2', 'fc_sc_unsigned_2_20_2', 'fc_u_unsigned_2_20_2','fc_kn_unsigned_2_75_2', 'fc_sc_unsigned_2_75_2', 'fc_u_unsigned_2_75_2', 'fc_kn_unsigned_2_100_2', 'fc_sc_unsigned_2_100_2', 'fc_u_unsigned_2_100_2', 'fc_kn_unsigned_2_50_50_2', 'fc_sc_unsigned_2_50_50_2', 'fc_u_unsigned_2_50_50_2']
    moons_data = [moons_data[architecture] for architecture in architecture_order]
    circles_data = get_data("make_circles", ["fc_kn_unsigned", "fc_sc_unsigned", "fc_u_unsigned"], results_path)
    circles_data = [circles_data[architecture] for architecture in architecture_order]
    digits_data = get_data("load_digits", ["fc_kn_unsigned", "fc_sc_unsigned", "fc_u_unsigned"], results_path)
    digits_data = list(digits_data.values())
    digits_binary_data = get_data("load_digits_binary",["fc_kn_unsigned", "fc_sc_unsigned", "fc_u_unsigned"], results_path)
    digits_binary_data = list(digits_binary_data.values())
    x_labels = ["", "A", "", "", "B", "", "", "C", "", "", "D", "", ]
    # Create Edge_popup_performance_comparison_boxplot_make_moons
    create_accuracy_box_plot("make_moons", moons_data, x_labels, grid=False)
    # Create Edge_popup_performance_comparison_boxplot_make_circles
    create_accuracy_box_plot("make_circles", circles_data, x_labels, grid=False)
    # Create Edge_popup_performance_comparison_boxplot_load_digits
    create_accuracy_box_plot("load_digits", digits_data, x_labels)
    # Create Edge_popup_performance_comparison_boxplot_load_digits_binary
    create_accuracy_box_plot("load_digits_binary", digits_binary_data, x_labels)

    # print(get_median_accuracies("make_moons"))
    # print(get_mean_accuracies("make_circles"))
    #sanity_check("load_digits")

    # Create make_circles mean accuracy plot with varying prune rates and GA comparison points
    considered_GA_config_make_moons = ["max_acc_bound", "max_acc_bound", "max_acc_bound", "top_speed"]
    x_labels = ["A,\n73.2%", "B,\n59.2%", "C,\n56.1%", "D,\n54.6%"]
    create_accuracy_box_plot_varied_prune_rates("make_moons", considered_GA_config_make_moons, x_labels,
                                                "varied_prune_rates", grid=True)

    # Create make_circles mean accuracy plot with varying prune rates and GA comparison points
    considered_GA_config_make_circles = ["top_speed", "top_speed", "max_acc_bound", "top_speed"]
    x_labels = ["A,\n66.3%", "B,\n62.3%", "C,\n53.9%", "D,\n58.1%"]
    create_accuracy_box_plot_varied_prune_rates("make_circles", considered_GA_config_make_circles, x_labels,
                                                "varied_prune_rates", grid=True)


    # Create blobs plots
    evaluate = {"model_args": {"nn_architecture": [2, 100], "init": "uniform", "prune_rate": 0.5, "activation": "relu",
                               "mask_except": ["net.0.bias", "net.2.bias"], "device": "cpu"},
                "dataset_args": {"name": "make_blobs", "norm": False, "device": "cpu"}}
    data = get_load_digits_fewer_classes_make_blobs_data_fc_u_unsigned("make_blobs", results_path)
    calculate_statistics_for_load_digits_fewer_classes_and_make_blobs_dataset("make_blobs", False,
                                                                              training_accuracy=False,
                                                                              evaluate=evaluate, source="tmp_log",
                                                                              data=data)

    create_accuracy_scatter_plot_make_blobs_load_digits_fewer_classes("make_blobs", False, data,
                                                                      grid=False, evaluate=evaluate)
    create_accuracy_box_plot_make_blobs_load_digits_fewer_classes("make_blobs", False, data, grid=True,
                                                                  evaluate=evaluate)

    # Create_make_blobs_load_digits_fewer_classes_combined_error_plot("load_digits_fewer_classes", True, True, with_additional_GA=True)
    data = get_load_digits_fewer_classes_make_blobs_data_fc_u_unsigned("load_digits_fewer_classes", results_path)
    calculate_statistics_for_load_digits_fewer_classes_and_make_blobs_dataset("load_digits_fewer_classes", False, training_accuracy=False, evaluate=None, source="tmp_log", data=data)


