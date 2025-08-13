import torch

from genetic_algorithm.methods.evaluation.base_individual_evaluation import IndividualEvaluationMethod


class IndividualPerformanceEvaluation(IndividualEvaluationMethod):
    """Evaluates individuals based on performance objective."""

    def __init__(self, quality_metric, model=None, train_loader=None, device=None):
        super(IndividualPerformanceEvaluation, self).__init__(quality_metric=quality_metric,
                                                              model=model, train_loader=train_loader, device=device)
        self.quality_metric = quality_metric
        self.model = model
        self.device = device
        self.train_loader = train_loader

    def evaluate_individual(self, bit_vector, evaluation_cache):
        """Evaluates an individual based on a neural network and a fitness function."""
        if evaluation_cache and evaluation_cache.has_evaluated(bit_vector):
            return evaluation_cache.retrieve_fitness(bit_vector)

        subnet = self.model.apply_mask(bit_vector)
        X = self.train_loader.dataset.tensors[0]
        y = self.train_loader.dataset.tensors[1]
        fitness = self.quality_metric(subnet(X), y, self.device)

        """subnet.eval()
        batch_fitness = []
        with torch.no_grad():
            for step, batch in enumerate(self.train_loader):
                batch = [tensor.to(self.device) for tensor in batch]
                batch_fitness.append(self.quality_metric(subnet(batch[0]), batch[1], self.device))

        fitness = np.mean(batch_fitness)"""

        if isinstance(fitness, torch.Tensor) and fitness.numel() == 1:
            fitness = fitness.item()

        if evaluation_cache:
            evaluation_cache.store_fitness(bit_vector, fitness)
        return fitness