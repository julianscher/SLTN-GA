from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from pandas import DataFrame
from sklearn.datasets import make_blobs
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import Dataset
from data.utils import normalize_dataset_over_interval
from utilities.helper_functions import get_results_path
from utilities.net_utils import predict


class BLOBS(Dataset):
    def __init__(self, centers, norm, device):
        super(BLOBS, self).__init__()
        self.centers = centers
        self.norm = norm

        X, Y = make_blobs(n_samples=5000*centers, centers=centers, random_state=10, center_box=(-20.0, 40.0))

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, shuffle=False
        )

        # Train dataset
        X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
        X_train_norm = normalize_dataset_over_interval(X_train, -0.7, 0.7)
        Y_train = torch.from_numpy(Y_train).to(torch.float32).to(device)
        self.train_set = SimpleNamespace(**{"X": X_train_norm if norm else X_train, "y": Y_train})
        self.train_loader = DataLoader(TensorDataset(X_train_norm if norm else X_train, Y_train),
                                       batch_size=500, shuffle=False, pin_memory=True)

        X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
        X_test_norm = normalize_dataset_over_interval(X_test, -0.7, 0.7)
        Y_test = torch.from_numpy(Y_test).to(torch.float32)
        self.test_set = SimpleNamespace(**{"X": X_test_norm if norm else X_test, "y": Y_test})
        self.test_loader = DataLoader(TensorDataset(X_test_norm if norm else X_test, Y_test),
                                      batch_size=500, shuffle=False, pin_memory=True)

    def plot_dataset(self, train_dataset, **kwargs):
        """ scatter plot, dots colored by class value """
        X, y = (self.train_set.X, self.train_set.y) if train_dataset else (self.test_set.X, self.test_set.y)
        df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
        colors = {idx: colors_list[idx] for idx in range(self.centers)}
        grouped = df.groupby('label')
        for idx, (key, group) in enumerate(grouped):
            plt.scatter(group['x'], group['y'], c=colors[idx], label=key, edgecolor="white", linewidth=1)
        plt.xlabel("$x_{0}$")
        plt.ylabel("$x_{1}$")
        plt.savefig(f"{get_results_path()}/blobs{'_norm_' if self.norm else ''}_{self.centers}centers.png")
        plt.show()

    def plot_learned_function(self, train_dataset, subnet, **kwargs):
        log_path = kwargs.get("log_path", get_results_path())
        X, y = (self.train_set.X, self.train_set.y) if train_dataset else (self.test_set.X, self.test_set.y)
        num_classes = self.centers

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
        cmap = mcolors.ListedColormap(colors[:num_classes])

        # Create meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        # Predict over meshgrid
        x_in = np.c_[xx.ravel(), yy.ravel()]
        x_in = torch.from_numpy(x_in).type(torch.float32)
        with torch.no_grad():
            y_pred = predict(subnet(x_in))
        y_pred = np.round(y_pred).reshape(xx.shape)

        # Colour areas
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, y_pred, levels=np.arange(-0.5, num_classes + 0.5, 1), cmap=cmap, alpha=0.3)

        # Draw blobs
        for i, color in enumerate(colors[:num_classes]):
            plt.scatter(X[y == i, 0], X[y == i, 1], color=color, edgecolor='black', label=f'Class {i}', s=30)

        plt.xlabel("$x_{0}$", fontsize=16)  # Tex format only works when saving the image
        plt.ylabel("$x_{1}$", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{log_path}/decision_boundary.png")
        plt.show()
        plt.clf()


