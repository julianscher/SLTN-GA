from genetic_algorithm.ga_configuration.ga_component_factory import GAComponentFactory
from genetic_algorithm.ga_configuration.ga_configuration import GAConfiguration


class GAConfigFactory:
    """Factory to create a GA configuration."""

    @staticmethod
    def create_ga_config(config: dict, worker_instance):
        factory = GAComponentFactory(worker_instance)
        generation = factory.create_generation(config["generation"]["method"], **config["generation"].get("args", {}))
        selection = factory.create_selection(config["selection"]["method"], **config["selection"].get("args", {}))
        mutation = factory.create_mutation(config["mutation"]["method"], **config["mutation"].get("args", {}))
        recombination = factory.create_recombination(config["recombination"]["method"], **config["recombination"].get("args", {}))
        survivor_evaluation = factory.create_survivor_evaluation(config["survivor_evaluation"]["method"], **config["survivor_evaluation"].get("args", {}))
        parents_evaluation = factory.create_parents_evaluation(config["parents_evaluation"]["method"], **config["parents_evaluation"].get("args", {}))
        individual_evaluation = factory.create_individual_evaluation(config["individual_evaluation"]["method"], **config["individual_evaluation"].get("args", {}))
        evaluation_cache = factory.create_evaluation_cache(config["evaluation_cache"]["type"], **config["evaluation_cache"].get("args", {}))

        return GAConfiguration(
            generation=generation,
            selection=selection,
            mutation=mutation,
            recombination=recombination,
            survivor_evaluation=survivor_evaluation,
            parents_evaluation=parents_evaluation,
            individual_evaluation=individual_evaluation,
            evaluation_cache=evaluation_cache
        )
