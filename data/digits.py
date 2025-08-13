import random
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import Dataset
from data.utils import normalize_dataset_over_interval
from utilities.helper_functions import get_results_path


class DIGITS(Dataset):
    def __init__(self, norm, device, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], with_validation=False):
        super(DIGITS, self).__init__()

        digits = load_digits()
        print(classes)

        # flatten the images
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        target = digits.target

        # data and target are sorted, i.e. an image at index x in data has its target (class) value at index x in target
        data = np.array([image for idx, image in enumerate(data) if target[idx] in classes])
        target = np.array([cls for cls in target if cls in classes])

        # Print how many samples there are for each class
        cls_indices = {t_cls: [idx for idx, cls in enumerate(target) if cls == t_cls] for t_cls in classes}
        [print(f"#samples (training + testing) for class {t_cls}:" if not with_validation else
               f"#samples (training + testing + validation) for class {t_cls}:", len(cls_indices[t_cls])) for t_cls in
         cls_indices.keys()]

        X_train, X_test, Y_train, Y_test = train_test_split(
            data, target, train_size=0.75, test_size=0.25, random_state=1
        )

        if with_validation:
            X_test, X_val, Y_test, Y_val = train_test_split(
                X_test, Y_test, train_size=0.5, test_size=0.5, random_state=1
            )

        # Train dataset
        X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
        X_train_norm = normalize_dataset_over_interval(X_train, -0.7, 0.7)
        Y_train = torch.from_numpy(Y_train).to(torch.float32).to(device)
        self.train_set = SimpleNamespace(**{"X": X_train_norm if norm else X_train, "y": Y_train})
        self.train_loader = DataLoader(TensorDataset(X_train_norm if norm else X_train, Y_train),
                                       batch_size=250, shuffle=False, pin_memory=True)

        # Test dataset
        X_test = torch.from_numpy(X_test).to(torch.float32).to(device)
        X_test_norm = normalize_dataset_over_interval(X_test, -0.7, 0.7)
        Y_test = torch.from_numpy(Y_test).to(torch.float32)
        self.test_set = SimpleNamespace(**{"X": X_test_norm if norm else X_test, "y": Y_test})
        self.test_loader = DataLoader(TensorDataset(X_test_norm if norm else X_test, Y_test),
                                      batch_size=20 if not with_validation else 10, shuffle=False, pin_memory=True)

        if with_validation:
            # Test dataset
            X_val = torch.from_numpy(X_val).to(torch.float32).to(device)
            X_val_norm = normalize_dataset_over_interval(X_val, -0.7, 0.7)
            Y_val = torch.from_numpy(Y_val).to(torch.float32)
            self.val_set = SimpleNamespace(**{"X": X_val_norm if norm else X_val, "y": Y_val})
            self.val_loader = DataLoader(TensorDataset(X_val_norm if norm else X_val, Y_val),
                                          batch_size=10, shuffle=False, pin_memory=True)

    def plot_random_digit(self):
        digits = load_digits()
        random_idx = random.choice(range(len(digits)))
        plt.imshow(digits.images[random_idx], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.savefig(f"{get_results_path()}/Random_Digit.png")
        plt.show()

    def plot_digit(self, index):
        digits = load_digits()
        plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.savefig(f"{get_results_path()}/Index{index}_Digit.png")
        plt.show()

    def get_digits_cls_indices(self):
        digits = load_digits()
        target = digits.target
        return {t_cls: [idx for idx, cls in enumerate(target) if cls == t_cls] for t_cls in digits.target_names}


