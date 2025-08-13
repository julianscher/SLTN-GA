from genetic_algorithm.methods.evaluation.fc import *
from genetic_algorithm.methods.evaluation.fc.hashing import FitnessHashTable
from genetic_algorithm.methods.generation.fc import *
from genetic_algorithm.methods.mutation.fc import *
from genetic_algorithm.methods.recombination.fc import *
from genetic_algorithm.methods.selection import *
from utilities.fc.fc_utils import get_NN_architecture_from_model
from utilities.parser import get_quality_metric


class GAComponentFactory:
    """Factory to create selection, crossover, and mutation methods based on a configuration."""

    _METHODS = {
        "fc": {
            "generation": {
                "from_individual_generation": FromIndividualGeneration,
                "lax_connectivity_alternative_faster_generation": LaxConnectivityAlternativeFasterGeneration,
                "lax_connectivity_alternative_generation": LaxConnectivityAlternativeGeneration,
                "lax_connectivity_alternative_with_upper_bound_generation": LaxConnectivityAlternativeWithUpperBoundGeneration,
                "lax_connectivity_generation": LaxConnectivityGeneration,
                "lax_connectivity_learned_accuracy_filter_generation": LaxConnectivityLearnedAccuracyFilterGeneration,
                "max_bit_vector_generation": MaxBitVectorGeneration,
                "negligent_generation": NegligentGeneration,
                "soft_connectivity_generation": SoftConnectivityGeneration,
                "strict_connectivity_generation": StrictConnectivityGeneration,
            },
            "selection": {
                "cutoff_selection": CutoffSelection,
                "graceful_cutoff_selection": GracefulCutoffSelection,
                "random_walk_selection": RandomWalkSelection,
                "roulette_selection": RouletteSelection,
            },
            "mutation": {
                "neuron_mutation": NeuronMutation,
                "path_mutation": PathMutation,
                "path_flip_mutation": PathFlipMutation,
                "single_point_mutation": SinglePointMutation,
                "single_point_negligent_optimized_mutation": SinglePointNegligentOptimizedMutation,
                "single_point_negligent_optimized_sourced_mutation": SinglePointNegligentOptimizedSourcedMutation,
                "single_point_path_fixed_mutation": SinglePointPathFixedMutation,
            },
            "recombination": {
                "fixed_path_crossover_recombination": FixedPathCrossoverRecombination,
                "input_output_path_crossover_recombination": InputOutputPathCrossoverRecombination,
                "neuron_crossover_recombination": NeuronCrossoverRecombination,
                "random_point_crossover_recombination": RandomPointCrossoverRecombination,
                "random_point_crossover_negligent_recombination": RandomPointCrossoverNegligentRecombination,
            },
            "population_evaluation": {
                "population_performance_evaluation": PopulationPerformanceEvaluation,
                "population_ranking_evaluation": PopulationRankingEvaluation,
            },
            "individual_evaluation": {
                "individual_performance_evaluation": IndividualPerformanceEvaluation,
            },
            "evaluation_cache": {
                "fitness_hash_table": FitnessHashTable,
            }
        },
    }

    _EVALUATION_CACHE_TYPES = {"fitness_hash_table": FitnessHashTable}

    def __init__(self, worker_instance):
        self.worker_instance = worker_instance
        self.model_type = self._detect_model_type()

    def create_component(self, method, component_type, **kwargs):
        """Creates a GA component dynamically based on model type."""
        self._append_auxiliary_arguments(kwargs)
        method_map = self._METHODS[self.model_type].get(component_type, {})
        if method not in method_map:
            raise ValueError(f"Unknown {component_type} method: {method}")
        return method_map[method](**kwargs)

    def create_generation(self, method, **kwargs):
        return self.create_component(method, "generation", **kwargs)

    def create_selection(self, method, **kwargs):
        return self.create_component(method, "selection", **kwargs)

    def create_mutation(self, method, **kwargs):
        return self.create_component(method, "mutation", **kwargs)

    def create_recombination(self, method, **kwargs):
        return self.create_component(method, "recombination", **kwargs)

    def create_survivor_evaluation(self, method, **kwargs):
        return self.create_component(method, "population_evaluation", **kwargs)

    def create_parents_evaluation(self, method, **kwargs):
        return self.create_component(method, "population_evaluation", **kwargs)

    def create_individual_evaluation(self, method, **kwargs):
        return self.create_component(method, "individual_evaluation", **kwargs)

    def create_evaluation_cache(self, cache_type, **kwargs):
        return self.create_component(cache_type, "evaluation_cache", **kwargs)

    def _detect_model_type(self):
        """Detects if the model is FC or something else based on batch format."""
        if self.worker_instance.model.__class__.__name__ in ["SubnetMLP", "SubnetMLPEmbedded", "SubnetSine"]:
            return "fc"
        else:
            raise(ValueError("Model type not supported."))

    def _append_auxiliary_arguments(self, kwargs):
        """Adds necessary arguments from worker_instance to kwargs."""
        worker = self.worker_instance
        data = worker.data

        aux_args = {
            "model": worker.model,
            "train_loader": data.train_loader,
            "test_loader": data.test_loader,
            "device": worker.device,
            "nn_architecture": get_NN_architecture_from_model(worker.model.net),
        }

        if "quality_metric" in kwargs:
            aux_args["quality_metric"] = get_quality_metric(kwargs["quality_metric"])

        kwargs.update({k: v for k, v in aux_args.items() if k in kwargs})



