import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from src.neural_network.base_networks import NN
from src.neural_network.helper_functions import path


class NNFramework:
    def __init__(self, NN_architecture):
        self.model = NN(NN_architecture)

    def train(self, dataloader, num_epochs, learning_rate):
        losses = []
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            best_acc = 0
            for it, (X_batch, Y_batch) in enumerate(dataloader):
                # Forward
                outp = self.model(X_batch)
                if outp.shape[1] > 1:
                    loss = nn.CrossEntropyLoss()(outp, Y_batch.long())
                else:
                    loss = nn.BCEWithLogitsLoss()(outp.flatten().flatten(), Y_batch)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                losses.append(loss.detach().flatten()[0])

                # Gradient descent or adam step
                optimizer.step()

                if outp.shape[1] > 1:
                    _, preds = outp.topk(1, 1, True, True)
                else:
                    probs = torch.sigmoid(outp)
                    preds = (probs > 0.5).type(torch.long)

                accuracy = accuracy_score(Y_batch, preds)
                if accuracy_score(Y_batch, preds) > best_acc:
                    best_acc = accuracy
            print(f"Best batch accuracy in Epoch {epoch}:", best_acc)

    @torch.no_grad()
    def predict(self, X):
        """ X: a set of samples, predictions: the set of labels that were predicted for X """
        self.model.eval()
        predictions = np.array([])
        outp = self.model(X)
        if outp.shape[1] > 1:
            _, preds = outp.topk(1, 1, True, True)
        else:
            probs = torch.sigmoid(outp)
            preds = (probs > 0.5).type(torch.long)
        predictions = np.hstack((predictions, preds.numpy().flatten()))
        return predictions.flatten()

    def get_accuracy(self, X, Y):
        """ X: a set of samples, Y: the corresponding true labels """
        return accuracy_score(Y, self.predict(X))

    @torch.no_grad()
    def get_loss(self, X, Y):
        """
        X: a set of samples
        Y: the corresponding true labels
        """
        outp = self.model(X)
        if outp.shape[1] > 1:
            return nn.CrossEntropyLoss()(self.model(X), Y.long())
        else:
            return nn.BCEWithLogitsLoss()(self.model(X).flatten(), Y)

    def get_number_of_parameters(self):
        params = []
        for name, parameter in self.model.named_parameters():
            if "score" not in name:
                params.append(parameter)
        return sum(p.numel() for p in params)

    def save(self):
        # save model parameters
        torch.save(self.model.state_dict(), f"{path}/model_parameters.pth")

