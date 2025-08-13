from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import Dataset
from data.utils import normalize_dataset_over_interval
from utilities.helper_functions import get_results_path
from utilities.net_utils import predict


class MOONS(Dataset):
    def __init__(self, norm, device):
        super(MOONS, self).__init__()
        self.norm = norm

        X, Y = make_moons(n_samples=66000, random_state=42, noise=0.07)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=16000, shuffle=False
        )

        # Train dataset
        X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
        X_train_norm = normalize_dataset_over_interval(X_train, -0.7, 0.7)
        Y_train = torch.from_numpy(Y_train).to(torch.float32).to(device)
        self.train_set = SimpleNamespace(**{"X": X_train_norm if norm else X_train, "y": Y_train})
        self.train_loader = DataLoader(TensorDataset(X_train_norm if norm else X_train, Y_train),
                                       batch_size=64, shuffle=False, pin_memory=True)

        X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
        X_test_norm = normalize_dataset_over_interval(X_test, -0.7, 0.7)
        Y_test = torch.from_numpy(Y_test).to(torch.float32)
        self.test_set = SimpleNamespace(**{"X": X_test_norm if norm else X_test, "y": Y_test})
        self.test_loader = DataLoader(TensorDataset(X_test_norm if norm else X_test, Y_test),
                                      batch_size=500, shuffle=False, pin_memory=True)

    def plot_dataset(self, train_dataset, **kwargs):
        X, y = (self.train_set.X, self.train_set.y) if train_dataset else (self.test_set.X, self.test_set.y)
        plt.figure(figsize=(8, 5))
        plt.scatter(X[100:, 0], X[100:, 1], c=y[100:], s=50,
                    cmap="RdBu", vmin=-.2, vmax=1.2,
                    edgecolor="white", linewidth=1)
        plt.xlabel("$x_{0}$")
        plt.ylabel("$x_{1}$")
        plt.savefig(f"{get_results_path()}/moons{'_norm_' if self.norm else ''}_RdBu.png")
        plt.show()
        plt.clf()

    def plot_learned_function(self, train_dataset, subnet, **kwargs):
        """
        X: a set of samples
        y: the corresponding true labels
        y_pred: the predicted labels
        inspired by https://psrivasin.medium.com/plotting-decision-boundaries-using-numpy-and-matplotlib-f5613d8acd19
        and https://www.kaggle.com/code/glebbuzin/solving-sklearn-datasets-with-pytorch
        """
        log_path = kwargs.get("log_path", get_results_path())
        X, y = (self.train_set.X, self.train_set.y) if train_dataset else (self.test_set.X, self.test_set.y)

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        x_in = np.c_[xx.ravel(), yy.ravel()]
        x_in = torch.from_numpy(x_in).type(torch.float32)
        with torch.no_grad():
            y_pred = predict(subnet(x_in))
        y_pred = np.round(y_pred).reshape(xx.shape)
        f, ax = plt.subplots(figsize=(13, 10))
        ax.set_title("Decision boundary", fontsize=20)
        contour = ax.contourf(xx, yy, y_pred, 25, cmap="RdBu",
                              vmin=0, vmax=1)

        ax.scatter(X[100:, 0], X[100:, 1], c=y[100:], s=50,
                   cmap="RdBu", vmin=-.2, vmax=1.2,
                   edgecolor="white", linewidth=1)

        plt.xlabel("$x_{0}$", fontsize=16) # Tex format only works when saving the image
        plt.ylabel("$x_{1}$", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{log_path}/decision_boundary.png")
        plt.show()
        plt.clf()


