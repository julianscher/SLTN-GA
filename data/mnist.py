import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from torchvision import datasets, transforms
from types import SimpleNamespace

from data.dataset import Dataset


class MNIST(Dataset):
    def __init__(self, norm, device, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        super(MNIST, self).__init__()
        self.norm = norm
        self.device = device
        self.batch_size = 128

        print(classes)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if norm else transforms.ToTensor()
        ])

        full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        full_dataset_indices = [i for i, label in enumerate(full_dataset.targets) if label in classes]
        test_dataset_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]

        full_data = torch.stack([full_dataset[i][0] for i in full_dataset_indices])
        full_labels = torch.tensor([full_dataset[i][1] for i in full_dataset_indices], dtype=torch.long)

        filtered_full_dataset = TensorDataset(full_data, full_labels)

        train_size = int(0.8 * len(filtered_full_dataset))
        val_size = len(filtered_full_dataset) - train_size
        train_subset, val_subset = random_split(filtered_full_dataset, [train_size, val_size])

        # Convert subsets back into TensorDatasets for data_loader.dataset.tensors access
        train_data = torch.stack([filtered_full_dataset[i][0] for i in train_subset.indices])
        train_data = train_data.view(train_data.shape[0], 784)
        train_labels = torch.tensor([filtered_full_dataset[i][1] for i in train_subset.indices], dtype=torch.long)
        train_dataset = TensorDataset(train_data, train_labels)

        val_data = torch.stack([filtered_full_dataset[i][0] for i in val_subset.indices])
        val_data = val_data.view(val_data.shape[0], 784)
        val_labels = torch.tensor([filtered_full_dataset[i][1] for i in val_subset.indices], dtype=torch.long)
        val_dataset = TensorDataset(val_data, val_labels)

        test_data = torch.stack([test_dataset[i][0] for i in test_dataset_indices])
        test_data = test_data.view(test_data.shape[0], 784)
        test_labels = torch.tensor([test_dataset[i][1] for i in test_dataset_indices], dtype=torch.long)
        filtered_test_dataset = TensorDataset(test_data, test_labels)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(filtered_test_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_set = SimpleNamespace(X=train_data, y=train_labels)
        self.val_set = SimpleNamespace(X=val_data, y=val_labels)
        self.test_set = SimpleNamespace(X=test_data, y=test_labels)

        print(f"Dataset initialized: {train_size} training samples, {val_size} validation samples, {len(filtered_test_dataset)} test samples.")