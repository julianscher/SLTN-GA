import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from data import MOONS, CIRCLES, DIGITS, BLOBS

# Define the parameter grids
param_grid_sgd = {
    'solver': ['sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
    'batch_size': [32, 64, 128],
    'momentum': [0.9, 0.95],
    'nesterovs_momentum': [True, False],
}

param_grid_sgd_randomized = {
    'solver': ['sgd'],
    'alpha': np.logspace(-4, -1, 10),
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': np.logspace(-3, -1, 10),
    'batch_size': [32, 64, 128],
    'momentum': np.linspace(0.0, 1.0, 5),
    'nesterovs_momentum': [True, False],
}

param_grid_adam = {
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
    'batch_size': [32, 64, 128],
    'epsilon': [1e-8, 1e-9]
}

param_grid_adam_randomized = {
    'solver': ['adam'],
    'alpha': np.logspace(-4, -1, 10),
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': np.logspace(-3, -1, 10),
    'batch_size': [32, 64, 128],
    'epsilon': np.logspace(-9, -7, 10)
}


def tune_hyperparameters(X_trainset, Y_trainset, X_testset, Y_testset, hidden_layer_sizes, param_grid):
    mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=hidden_layer_sizes)

    # Perform GridSearchCV
    grid_search = RandomizedSearchCV(mlp, param_grid, n_jobs=-1, cv=3, verbose=2, random_state=42)
    grid_search.fit(X_trainset, Y_trainset)

    # Print the best parameters and the best score
    print(f"Best parameters found for hidden_layer_sizes{hidden_layer_sizes}: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    # Evaluate on test data
    best_mlp = grid_search.best_estimator_
    test_score = best_mlp.score(X_testset, Y_testset)
    print(f"Test set score: {test_score}")


def tune_for_moons():
    # Load dataset
    dataset_name = "make_moons"
    data = MOONS(norm=True, device="cpu")
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y

    hidden_layer_sizes_list = [(20,), (75,), (100,), (50, 50)]
    for hidden_layer_sizes in hidden_layer_sizes_list:
        # Find optimal parameters
        tune_hyperparameters(X_trainset, Y_trainset, X_testset, Y_testset, hidden_layer_sizes, param_grid_adam_randomized)


def tune_for_circles():
    # Load dataset
    dataset_name = "make_circles"
    data = CIRCLES(norm=False, device="cpu")
    X_trainset = data.train_set.X
    Y_trainset = data.train_set.y
    X_testset = data.test_set.X
    Y_testset = data.test_set.y

    hidden_layer_sizes_list = [(20,), (75,), (100,), (50, 50)]
    for hidden_layer_sizes in hidden_layer_sizes_list:
        # Find optimal parameters
        tune_hyperparameters(X_trainset, Y_trainset, X_testset, Y_testset, hidden_layer_sizes, param_grid_adam_randomized)

def tune_for_digits(norm=True):
    dataset_names = ["load_digits_binary", "load_digits_ternary", "load_digits_quaternary", "load_digits_quinary", "load_digits_denary"]
    classes_list = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    device = "cpu"
    hidden_layer_sizes = (75,)
    for classes in classes_list:
        data = DIGITS(classes=classes, norm=norm, device=device)
        X_trainset = data.train_set.X
        Y_trainset = data.train_set.y
        X_testset = data.test_set.X
        Y_testset = data.test_set.y

        tune_hyperparameters(X_trainset, Y_trainset, X_testset, Y_testset, hidden_layer_sizes, param_grid_adam_randomized)


def tune_for_blobs(norm=False):
    dataset_names = ["make_two_blobs", "make_three_blobs", "make_four_blobs", "make_five_blobs", "make_six_blobs", "make_seven_blobs", "make_eight_blobs", "make_nine_blobs", "make_ten_blobs"]
    centers_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    device = "cpu"
    hidden_layer_sizes = (100,)
    for centers in centers_list:
        data = BLOBS(centers=centers, norm=norm, device=device)
        X_trainset = data.train_set.X
        Y_trainset = data.train_set.y
        X_testset = data.test_set.X
        Y_testset = data.test_set.y

        tune_hyperparameters(X_trainset, Y_trainset, X_testset, Y_testset, hidden_layer_sizes, param_grid_adam_randomized)



if __name__ == "__main__":
    tune_for_digits(False)
