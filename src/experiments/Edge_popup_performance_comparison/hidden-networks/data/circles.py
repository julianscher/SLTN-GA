import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import make_circles


class CIRCLES:
    def __init__(self, args):
        super(CIRCLES, self).__init__()

        use_cuda = False

        X_train, Y_train = make_circles(n_samples=50000, random_state=42, noise=0.07)
        X_train = torch.from_numpy(X_train).to(torch.float32)
        Y_train = torch.from_numpy(Y_train).to(torch.float32)

        X_test, Y_test = make_circles(n_samples=16000, random_state=42, noise=0.07)
        X_test = torch.from_numpy(X_test).to(torch.float32)
        Y_test = torch.from_numpy(Y_test).to(torch.float32)


        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=1000,
            shuffle=False,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            TensorDataset(X_test, Y_test),
            batch_size=1,  # GA doesn't use dataloader for evaluating the individuals. The accuracy is calculated directly on X_test_std and Y_test
            shuffle=False,
            **kwargs
        )