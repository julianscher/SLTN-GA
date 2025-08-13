import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.datasets import load_digits


def standardize_dataset_over_interval(dataset, a, b):
    return (b - a) * (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset)) + a


class DIGITS2:
    def __init__(self, args):
        super(DIGITS2, self).__init__()

        use_cuda = False

        digits = load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        target = digits.target

        # data and target are sorted, i.e. an image at index x in data has its target (class) value at index x in target
        data = np.array([image for idx, image in enumerate(data) if target[idx] in [0, 1]])
        target = np.array([cls for cls in target if cls in [0, 1]])

        # Print how many samples there are for each class
        cls_indices = {t_cls: [idx for idx, cls in enumerate(target) if cls == t_cls] for t_cls in [0, 1]}
        [print(f"#samples (training + testing) for class {t_cls}:", len(cls_indices[t_cls])) for t_cls in
         cls_indices.keys()]

        X_train, X_test, Y_train, Y_test = train_test_split(
            data, target, test_size=0.25, shuffle=False
        )

        X_train_std = standardize_dataset_over_interval(torch.from_numpy(X_train).to(torch.float32), -0.7, 0.7)
        Y_train = torch.from_numpy(Y_train).to(torch.float32)

        X_test_std = standardize_dataset_over_interval(torch.from_numpy(X_test).to(torch.float32), -0.7, 0.7)
        Y_test = torch.from_numpy(Y_test).to(torch.float32)

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            TensorDataset(X_train_std, Y_train),
            batch_size=250,
            shuffle=False,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            TensorDataset(X_test_std, Y_test),
            batch_size=1,
            # GA doesn't use dataloader for evaluating the individuals. The accuracy is calculated directly on X_test_std and Y_test
            shuffle=False,
            **kwargs
        )
