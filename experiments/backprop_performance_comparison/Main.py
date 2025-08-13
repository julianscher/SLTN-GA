import os
import time
import dill
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from data import MOONS, CIRCLES, DIGITS, BLOBS
from utilities.helper_functions import get_results_path

results_path = f"{get_results_path()}/backprop_performance_results"

# Experiment contents overview
NN_architectures_binary = [[2, 20, 2], [2, 75, 2], [2, 100, 2], [2, 50, 50, 2]]
NN_architectures_digits = [[64, 20, 10], [64, 75, 10], [64, 100, 10], [64, 50, 50, 10]]
NN_architectures_digits_binary = [[64, 20, 2], [64, 75, 2], [64, 100, 2], [64, 50, 50, 2]]

if not os.path.isdir(results_path):
    os.mkdir(results_path)


def create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset, dataset_name, runs):
    training_times = []
    final_accuracies = []

    # Train networks and collect data
    for i in range(runs):
        print(NN_architecture, i)
        start_time = time.time()
        hyperparameters['hidden_layer_sizes'] = tuple(NN_architecture[1:-1])
        hyperparameters['max_iter'] = 1000
        mlp = MLPClassifier(**hyperparameters)
        mlp.fit(X_trainset, Y_trainset)
        predictions = mlp.predict(X_testset)
        training_times.append(time.time() - start_time)
        acc_score = accuracy_score(Y_testset, predictions)
        final_accuracies.append(acc_score)
        print(i, acc_score)

    backprop_data = {"training_times": training_times, "final_accuracies": final_accuracies}

    # Save data
    if not os.path.isdir(f"{results_path}/{dataset_name}"):
        os.mkdir(f"{results_path}/{dataset_name}")
    with open(f"{results_path}/{dataset_name}/pickled_backprop_data_{NN_architecture}_{dataset_name}", "wb") as dill_file:
        dill.dump(backprop_data, dill_file)


def experiment_make_moons():
    dataset_name = "make_moons"
    device = "cpu"
    NN_architectures = NN_architectures_binary
    data = MOONS(norm=True, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    hyperparameters = [{'solver': 'adam', 'learning_rate_init': 0.021544, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 128, 'alpha': 0.0001},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215}]

    for idx, NN_architecture in enumerate(NN_architectures):
        create_backprop_data(NN_architecture, hyperparameters[idx], X_trainset, Y_trainset, X_testset, Y_testset,
                             dataset_name, runs=50)


def experiment_make_circles():
    dataset_name = "make_circles"
    device = "cpu"
    NN_architectures = NN_architectures_binary
    data = CIRCLES(norm=False, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    hyperparameters = [{'solver': 'sgd', 'nesterovs_momentum': False, 'momentum': 0.0,
                        'learning_rate_init': 0.1, 'learning_rate': 'adaptive',
                        'batch_size': 64, 'alpha': 0.046416},
                       {'solver': 'sgd', 'nesterovs_momentum': True, 'momentum': 0.5,
                        'learning_rate_init': 0.004642, 'learning_rate': 'adaptive',
                        'batch_size': 128, 'alpha': 0.046416},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215},
                       {'solver': 'sgd', 'nesterovs_momentum': True, 'momentum': 0.0,
                        'learning_rate_init': 0.1, 'learning_rate': 'adaptive',
                        'batch_size': 128, 'alpha': 0.046416}
                       ]

    for idx, NN_architecture in enumerate(NN_architectures):
        create_backprop_data(NN_architecture, hyperparameters[idx], X_trainset, Y_trainset, X_testset, Y_testset,
                             dataset_name, runs=50)

def experiment_load_digits_binary(norm=True):
    classes = [0, 1]
    dataset_name = "load_digits_binary"
    device = "cpu"
    NN_architecture = [64, 75, 2]
    data = DIGITS(classes=classes, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_load_digits_ternary(norm=True):
    classes = [0, 1, 2]
    dataset_name = "load_digits_ternary"
    device = "cpu"
    NN_architecture = [64, 75, 3]
    data = DIGITS(classes=classes, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.021544346900318832, 'learning_rate': 'constant',
                           'epsilon': 1.6681005372000556e-09, 'batch_size': 128, 'alpha': 0.046415888336127774}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.0027825594022071257, 'learning_rate': 'adaptive',
                           'epsilon': 5.99484250318941e-08, 'batch_size': 32, 'alpha': 0.01}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_load_digits_quaternary(norm=True):
    classes = [0, 1, 2, 3]
    dataset_name = "load_digits_quaternary"
    device = "cpu"
    NN_architecture = [64, 75, 4]
    data = DIGITS(classes=classes, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.0027825594022071257, 'learning_rate': 'adaptive',
                           'epsilon': 7.742636826811278e-09, 'batch_size': 32, 'alpha': 0.004641588833612777}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.0016681005372000592, 'learning_rate': 'adaptive',
                           'epsilon': 1e-07, 'batch_size': 64, 'alpha': 0.046415888336127774}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_load_digits_quinary(norm=True):
    classes = [0, 1, 2, 3, 4]
    dataset_name = "load_digits_quinary"
    device = "cpu"
    NN_architecture = [64, 75, 5]
    data = DIGITS(classes=classes, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.0027825594022071257, 'learning_rate': 'adaptive',
                           'epsilon': 5.99484250318941e-08, 'batch_size': 32, 'alpha': 0.01}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'adaptive',
                           'epsilon': 1e-07, 'batch_size': 128, 'alpha': 0.046415888336127774}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_load_digits_denary(norm=True):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_name = "load_digits_denary"
    device = "cpu"
    NN_architecture = [64, 75, 10]
    data = DIGITS(classes=classes, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.0027825594022071257, 'learning_rate': 'adaptive',
                           'epsilon': 7.742636826811278e-09, 'batch_size': 32, 'alpha': 0.004641588833612777}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.0027825594022071257, 'learning_rate': 'adaptive',
                           'epsilon': 7.742636826811278e-09, 'batch_size': 32, 'alpha': 0.004641588833612777}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_two_blobs(norm=False):
    centers = 2
    dataset_name = "make_two_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 2]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_three_blobs(norm=False):
    centers = 2
    dataset_name = "make_three_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 3]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_four_blobs(norm=False):
    centers = 4
    dataset_name = "make_four_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 4]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)


def experiment_make_five_blobs(norm=False):
    centers = 5
    dataset_name = "make_five_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 5]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters =  {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                            'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_six_blobs(norm=False):
    centers = 6
    dataset_name = "make_six_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 6]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_seven_blobs(norm=False):
    centers = 7
    dataset_name = "make_seven_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 7]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_eight_blobs(norm=False):
    centers = 8
    dataset_name = "make_eight_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 8]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)

def experiment_make_nine_blobs(norm=False):
    centers = 9
    dataset_name = "make_nine_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 9]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)


def experiment_make_ten_blobs(norm=False):
    centers = 10
    dataset_name = "make_ten_blobs"
    device = "cpu"
    NN_architecture = [2, 100, 10]
    data = BLOBS(centers=centers, norm=norm, device=device)
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y
    if norm:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}
    else:
        hyperparameters = {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                           'epsilon': 4.641588833612773e-09, 'batch_size': 64, 'alpha': 0.00021544346900318845}

    create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset,
                         dataset_name, runs=25)




if __name__ == "__main__":
    experiment_load_digits_denary(False)