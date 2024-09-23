import os
import sys
import time
from pathlib import Path
import dill
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from src.neural_network.dataset import get_dataset

sys.path.append(str(Path(__file__).parent))
results_rel_path = "resources/test_results/Backprop_performance_results"
results_path = Path(os.path.join(Path(__file__).parent.parent.parent.parent, results_rel_path))

# Experiment contents overview
NN_architectures_binary = [[2, 20, 2], [2, 75, 2], [2, 100, 2], [2, 50, 50, 2]]
NN_architectures_digits = [[64, 20, 10], [64, 75, 10], [64, 100, 10], [64, 50, 50, 10]]
NN_architectures_digits_binary = [[64, 20, 2], [64, 75, 2], [64, 100, 2], [64, 50, 50, 2]]
data = ["make_moons", "make_circles", "load_digits", "load_digits_binary"]

if not os.path.isdir(results_path):
    os.mkdir(results_path)


def create_backprop_data(NN_architecture, hyperparameters, X_trainset, Y_trainset, X_testset, Y_testset, dataset_name):
    training_times = []
    final_accuracies = []

    # Train networks and collect data
    for i in range(50):
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


def experiment1():
    dataset_name = "make_moons"
    X_trainset = get_dataset(dataset_name)["X_train_std"]
    Y_trainset = get_dataset(dataset_name)["Y_train"]
    X_testset = get_dataset(dataset_name)["X_test_std"]
    Y_testset = get_dataset(dataset_name)["Y_test"]
    NN_architectures = NN_architectures_binary
    hyperparameters = [{'solver': 'adam', 'learning_rate_init': 0.021544, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 128, 'alpha': 0.0001},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215},
                       {'solver': 'adam', 'learning_rate_init': 0.001, 'learning_rate': 'constant',
                        'epsilon': 4.64159e-09, 'batch_size': 64, 'alpha': 0.000215}]
    return dataset_name, X_trainset, Y_trainset, X_testset, Y_testset, NN_architectures, hyperparameters


def experiment2():
    dataset_name = "make_circles"
    X_trainset = get_dataset(dataset_name)["X_train"]
    Y_trainset = get_dataset(dataset_name)["Y_train"]
    X_testset = get_dataset(dataset_name)["X_test"]
    Y_testset = get_dataset(dataset_name)["Y_test"]
    NN_architectures = NN_architectures_binary
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
    return dataset_name, X_trainset, Y_trainset, X_testset, Y_testset, NN_architectures, hyperparameters


def experiment3():
    dataset_name = "load_digits"
    X_trainset = get_dataset(dataset_name)["X_train_std"]
    Y_trainset = get_dataset(dataset_name)["Y_train"]
    X_testset = get_dataset(dataset_name)["X_test_std"]
    Y_testset = get_dataset(dataset_name)["Y_test"]
    NN_architectures = NN_architectures_digits
    hyperparameters = []
    return dataset_name, X_trainset, Y_trainset, X_testset, Y_testset, NN_architectures, hyperparameters

def experiment4():
    dataset_name = "load_digits_binary"
    X_trainset = get_dataset(dataset_name)["X_train_std"]
    Y_trainset = get_dataset(dataset_name)["Y_train"]
    X_testset = get_dataset(dataset_name)["X_test_std"]
    Y_testset = get_dataset(dataset_name)["Y_test"]
    NN_architectures = NN_architectures_digits_binary
    hyperparameters = []
    return dataset_name, X_trainset, Y_trainset, X_testset, Y_testset, NN_architectures, hyperparameters


if __name__ == "__main__":
    dataset_name, X_trainset, Y_trainset, X_testset, Y_testset, NN_architectures, hyperparameters = experiment1()
    for idx, NN_architecture in enumerate(NN_architectures):
        create_backprop_data(NN_architecture, hyperparameters[idx], X_trainset, Y_trainset, X_testset, Y_testset, dataset_name)