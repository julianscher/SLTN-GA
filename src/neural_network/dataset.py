import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles, load_digits, make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def standardize_dataset_over_interval(dataset, a, b):
    return (b - a) * (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset)) + a


def upscale_probs(probs):
    return ((probs - np.min(probs)) / (np.max(probs) - np.min(probs))) * (1 - 0) + 0


def get_digits_cls_indices():
    digits = load_digits()
    target = digits.target
    return {t_cls: [idx for idx, cls in enumerate(target) if cls == t_cls] for t_cls in digits.target_names}


def create_sklearn_make_moons_or_circles_dataset(generator, X_train_n_samples, X_train_random_state, X_train_noise, X_train_std_a, X_train_std_b,
                                                 train_dataloader_batch_size, train_dataloader_std_batch_size,
                                                 X_test_n_samples, X_test_random_state, X_test_noise, X_test_std_a, X_test_std_b):
    """ Using different values for X_train_n_samples and X_test_n_samples and a noise value >= 0.0 will result in
    largely different datasets even though the same random_state is used.
    But using the same random_state ensures that both datasets will still be sampled from the same distribution """

    sklearn_dataset = {}
    sklearn_dataset["X_train"] = torch.from_numpy(generator(n_samples=X_train_n_samples, random_state=X_train_random_state, noise=X_train_noise)[0]).to(torch.float32)
    sklearn_dataset["X_train_std"] = standardize_dataset_over_interval(sklearn_dataset["X_train"], X_train_std_a,X_train_std_b)
    sklearn_dataset["Y_train"] = torch.from_numpy(generator(n_samples=X_train_n_samples, random_state=X_train_random_state, noise=X_train_noise)[1]).to(torch.float32)
    sklearn_dataset["train_dataset"] = TensorDataset(sklearn_dataset["X_train"], sklearn_dataset["Y_train"])
    sklearn_dataset["train_dataset_std"] = TensorDataset(sklearn_dataset["X_train_std"],sklearn_dataset["Y_train"])
    sklearn_dataset["train_dataloader"] = DataLoader(sklearn_dataset["train_dataset"], batch_size=train_dataloader_batch_size)
    sklearn_dataset["train_dataloader_std"] = DataLoader(sklearn_dataset["train_dataset_std"],batch_size=train_dataloader_std_batch_size)
    sklearn_dataset["X_test"] = torch.from_numpy(generator(n_samples=X_test_n_samples, random_state=X_test_random_state, noise=X_test_noise)[0]).to(torch.float32)
    sklearn_dataset["X_test_std"] = standardize_dataset_over_interval(sklearn_dataset["X_test"], X_test_std_a,X_test_std_b)
    sklearn_dataset["Y_test"] = torch.from_numpy(generator(n_samples=X_test_n_samples, random_state=X_test_random_state, noise=X_test_noise)[1]).to(torch.float32)

    return sklearn_dataset


def create_sklearn_make_blobs_dataset(centers, center_box, n_samples, test_size, random_state, X_train_std_a, X_train_std_b,
                                      train_dataloader_batch_size, train_dataloader_std_batch_size, X_test_std_a,
                                      X_test_std_b):

    X, Y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state, center_box=center_box)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=False
    )

    sklearn_dataset = {}
    sklearn_dataset["X_train"] = torch.from_numpy(X_train).to(torch.float32)
    sklearn_dataset["X_train_std"] = standardize_dataset_over_interval(sklearn_dataset["X_train"], X_train_std_a,X_train_std_b)
    sklearn_dataset["Y_train"] = torch.from_numpy(Y_train).to(torch.float32)
    sklearn_dataset["train_dataset"] = TensorDataset(sklearn_dataset["X_train"], sklearn_dataset["Y_train"])
    sklearn_dataset["train_dataset_std"] = TensorDataset(sklearn_dataset["X_train_std"],sklearn_dataset["Y_train"])
    sklearn_dataset["train_dataloader"] = DataLoader(sklearn_dataset["train_dataset"], batch_size=train_dataloader_batch_size)
    sklearn_dataset["train_dataloader_std"] = DataLoader(sklearn_dataset["train_dataset_std"],batch_size=train_dataloader_std_batch_size)
    sklearn_dataset["X_test"] = torch.from_numpy(X_test).to(torch.float32)
    sklearn_dataset["X_test_std"] = standardize_dataset_over_interval(sklearn_dataset["X_test"], X_test_std_a,X_test_std_b)
    sklearn_dataset["Y_test"] = torch.from_numpy(Y_test).to(torch.float32)

    return sklearn_dataset


def load_sklearn_digits_dataset(test_size, X_train_std_a, X_train_std_b, train_dataloader_batch_size,
                                train_dataloader_std_batch_size, X_test_std_a, X_test_std_b,
                                classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    digits = load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    target = digits.target

    # data and target are sorted, i.e. an image at index x in data has its target (class) value at index x in target
    data = np.array([image for idx, image in enumerate(data) if target[idx] in classes])
    target = np.array([cls for cls in target if cls in classes])

    # Print how many samples there are for each class
    cls_indices = {t_cls: [idx for idx, cls in enumerate(target) if cls == t_cls] for t_cls in classes}
    [print(f"#samples (training + testing) for class {t_cls}:", len(cls_indices[t_cls])) for t_cls in cls_indices.keys()]

    X_train, X_test, Y_train, Y_test = train_test_split(
        data, target, test_size=test_size, shuffle=False
    )

    sklearn_dataset = {}
    sklearn_dataset["X_train"] = torch.from_numpy(X_train).to(torch.float32)
    sklearn_dataset["X_train_std"] = standardize_dataset_over_interval(sklearn_dataset["X_train"], X_train_std_a, X_train_std_b)
    sklearn_dataset["Y_train"] = torch.from_numpy(Y_train).to(torch.float32)
    sklearn_dataset["train_dataset"] = TensorDataset(sklearn_dataset["X_train"], sklearn_dataset["Y_train"])
    sklearn_dataset["train_dataset_std"] = TensorDataset(sklearn_dataset["X_train_std"], sklearn_dataset["Y_train"])
    sklearn_dataset["train_dataloader"] = DataLoader(sklearn_dataset["train_dataset"], batch_size=train_dataloader_batch_size)
    sklearn_dataset["train_dataloader_std"] = DataLoader(sklearn_dataset["train_dataset_std"], batch_size=train_dataloader_std_batch_size)
    sklearn_dataset["X_test"] = torch.from_numpy(X_test).to(torch.float32)
    sklearn_dataset["X_test_std"] = standardize_dataset_over_interval(sklearn_dataset["X_test"], X_test_std_a, X_test_std_b)
    sklearn_dataset["Y_test"] = torch.from_numpy(Y_test).to(torch.float32)

    return sklearn_dataset


def get_dataset(dataset_name):
    print("selected dataset: ", dataset_name)
    if dataset_name == "make_moons":
        return create_sklearn_make_moons_or_circles_dataset(make_moons, 50000, 42, 0.07, -0.7, 0.7, 500, 1000, 16000, 42, 0.07, -0.7, 0.7)
    elif dataset_name == "make_circles":
        return create_sklearn_make_moons_or_circles_dataset(make_circles, 50000, 42, 0.07, -0.7, 0.7, 500, 1000, 16000, 42, 0.07, -0.7, 0.7)
    elif dataset_name == "load_digits":
        return load_sklearn_digits_dataset(0.25, -0.7, 0.7, 250, 250, -0.7, 0.7)
    elif dataset_name in ["load_digits_binary", "load_digits_ternary", "load_digits_quaternary", "load_digits_quinary", "load_digits_senary", "load_digits_septenary", "load_digits_octonary", "load_digits_nonary"]:
        load_digits_fewer_classes = {"load_digits_binary": [0, 1], "load_digits_ternary": [0, 1, 2], "load_digits_quaternary": [0, 1, 2, 3], "load_digits_quinary": [0, 1, 2, 3, 4],
                                     "load_digits_senary": [0, 1, 2, 3, 4, 5], "load_digits_octonary": [0, 1, 2, 3, 4, 5, 6], "load_digits_nonary": [0, 1, 2, 3, 4, 5, 6, 7]}
        return load_sklearn_digits_dataset(0.25, -0.7, 0.7, 250, 250, -0.7, 0.7, load_digits_fewer_classes[dataset_name])
    elif dataset_name in ["make_two_blobs", "make_three_blobs", "make_four_blobs", "make_five_blobs", "make_six_blobs", "make_seven_blobs", "make_eight_blobs", "make_nine_blobs", "make_ten_blobs"]:
        make_blobs_centres_count = {"make_two_blobs": 2, "make_three_blobs": 3, "make_four_blobs": 4, "make_five_blobs": 5,
                                    "make_six_blobs": 6, "make_seven_blobs": 7, "make_eight_blobs": 8, "make_nine_blobs": 9, "make_ten_blobs": 10}
        centers = make_blobs_centres_count[dataset_name]
        return create_sklearn_make_blobs_dataset(centers, (-20.0, 40.0), 5000*centers, 0.25, 10, -0.7, 0.7, 500, 500, -0.7, 0.7)