import os

import torch
from torch import nn


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader=None, test_loader=None, device="cpu",
                 PATH=None, save_every=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.PATH = PATH
        self.save_every = save_every

    def _run_batch(self, X_batch, y_batch, optimize=True):
        output = self.model(X_batch)
        loss = nn.CrossEntropyLoss().to(self.device)(output, y_batch.long())
        if optimize:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return output, loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"{f'[GPU-{self.device}]' if self.device != 'cpu' else 'CPU'} Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True)
            self._run_batch(X_batch, y_batch)

    def _evaluate(self, loader, metric):
        self.model.eval()
        total_loss = 0
        total_metric = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output, loss = self._run_batch(X_batch, y_batch, optimize=False)
                total_loss += loss
                if metric:
                    total_metric += metric(output, y_batch, self.device)
        total_loss /= len(loader)
        total_metric = total_metric / len(loader) if metric else None
        return total_loss, total_metric

    def _save_checkpoint(self, epoch):
        ckp = self.model.net.state_dict()
        ckp_path = os.path.join(self.PATH, 'checkpoints')
        if not os.path.isdir(ckp_path):
            os.mkdir(ckp_path)
        ckp_path = os.path.join(ckp_path, f'epoch{epoch}.pt')
        torch.save(ckp, ckp_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {ckp_path}")

    def save_model(self):
        model_path = os.path.join(self.PATH, 'model.pt')
        torch.save(self.model.net.state_dict(), model_path)

    def train(self, num_epochs: int):
        print("Start training")
        self.model.train()
        self._save_checkpoint(0)
        for epoch in range(num_epochs):
            self._run_epoch(epoch)
            if self.save_every != 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    def validate(self, metric):
        print("Start validation")
        if self.val_loader is None:
            raise ValueError("Validation loader is not provided.")
        return self._evaluate(self.val_loader, metric)

    def test(self, metric):
        print("Start testing")
        if self.test_loader is None:
            raise ValueError("Test loader is not provided.")
        return self._evaluate(self.test_loader, metric)
