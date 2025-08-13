import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.datasets import make_moons, make_blobs


def normalize_dataset_over_interval(dataset, a, b):
    return (b - a) * (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset)) + a

class BLOBS:
    def __init__(self, args):
        super(BLOBS, self).__init__()
        self.centers = args.dataset_args.get("centers", 10)
        self.norm = args.dataset_args.get("norm", False)
        self.device = args.dataset_args.get("device", "cpu")

        use_cuda = False

        X, Y = make_blobs(n_samples=5000 * self.centers, centers=self.centers, random_state=10, center_box=(-20.0, 40.0))

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, shuffle=False
        )

        # Train dataset
        X_train = torch.from_numpy(X_train).to(torch.float32).to(self.device)
        X_train_norm = normalize_dataset_over_interval(X_train, -0.7, 0.7)
        Y_train = torch.from_numpy(Y_train).to(torch.float32).to(self.device)

        X_test = torch.from_numpy(X_test).to(torch.float32).to(self.device)
        X_test_norm = normalize_dataset_over_interval(X_test, -0.7, 0.7)
        Y_test = torch.from_numpy(Y_test).to(torch.float32).to(self.device)


        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            TensorDataset(X_train_norm if self.norm else X_train, Y_train),
            batch_size=500,
            shuffle=False,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            TensorDataset(X_test_norm if self.norm else X_test, Y_test),
            batch_size=1,  # GA doesn't use dataloader for evaluating the individuals. The accuracy is calculated directly on X_test_std and Y_test
            shuffle=False,
            **kwargs
        )
