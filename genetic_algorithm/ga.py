import os

import torch

from genetic_algorithm.population import Population


class GeneticAlgorithm:
    def __init__(self, model, criterion, train_loader, val_loader, test_loader, device, logger, population_config,
                 termination_condition, ga_config, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

        print("Generating population...")
        self.population = Population(**population_config, ga_config=ga_config)
        self.termination_condition = termination_condition
        self.args = args
        self.logger = logger

    def evolve(self):
        print("Start evolution")
        while not self._terminate():
            self.population.evolve()
            self.logger.apply_evol_routines(self)

        return self.population

    def validate_top_individual(self, metric):
        print("Start validation")
        if self.val_loader is None:
            raise ValueError("Validation loader is not provided.")
        return self._evaluate(self.val_loader, metric)

    def test_top_individual(self, metric):
        print("Start testing")
        if self.test_loader is None:
            raise ValueError("Test loader is not provided.")
        return self._evaluate(self.test_loader, metric)

    def _run_batch(self, subnet, X_batch, y_batch):
        output = subnet(X_batch)
        loss = self.criterion(output, y_batch, self.device)
        return output, loss.item()

    def _evaluate(self, loader, metric):
        subnet = self.best_model

        subnet.eval()
        total_loss = 0
        total_metric = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output, loss = self._run_batch(subnet, X_batch, y_batch)
                total_loss += loss
                if metric:
                    total_metric += metric(output, y_batch, self.device)
        total_loss /= len(loader)
        total_metric = total_metric / len(loader) if metric else None
        return total_loss, total_metric

    def _terminate(self):
        """ Checks if population fulfills condition for termination. """
        if self.termination_condition.method(self, self.logger.tmp_log, self.termination_condition.args):
            return True
        return False

    def best_model(self):
        top_individual = self.population.individuals[0].genome
        subnet = self.model.apply_mask(top_individual)
        return subnet

    def save_model(self):
        model_path = os.path.join(self.logger.log_path, 'model.pt')
        model = self.best_model
        torch.save(model.net.state_dict(), model_path)