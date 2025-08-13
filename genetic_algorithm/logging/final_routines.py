import os
import time
from copy import deepcopy

import dill
import numpy as np
import torch
from matplotlib import pyplot as plt

from data import MOONS
from genetic_algorithm.logging.evol_routines import er13, er7, er8, er10, er9, er6, er14, er15
from genetic_algorithm.logging.init_routines import ir1, ir11, ir5, ir6, ir8, ir7, ir9, ir12, ir14, ir15
from genetic_algorithm.population import Population
from utilities.fc.fc_utils import get_NN_architecture_from_model
from utilities.ga_utils import get_leaves_dfs, get_genealogy_bfs
from utilities.fc.fc_subnetwork import get_subnetwork
from utilities.net_utils import accuracy


def fr1(ga, tmp_log):
    duplicates = 0
    for occurrence in tmp_log["genomes_occurrences"].values():
        if occurrence > 1:
            duplicates += occurrence - 1

    print("duplicates:", duplicates)
    print("number_of_different_genomes:", len(tmp_log["genomes_occurrences"]))

    ga.population.evaluate_survivors()
    print(ga.population.__str__())

    print("--- %s seconds ---" % (time.time() - tmp_log["start_time"]))


def fr2(ga, tmp_log):
    plt.figure(figsize=(9, 6))
    losses = tmp_log["loss_development"].values()
    generations = tmp_log["loss_development"].keys()
    plt.plot(generations, losses)
    plt.title("loss development", fontsize=20)
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("loss of best individual", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/loss_development")
    else:
        plt.show()
    plt.clf()


def fr3(ga, tmp_log):
    plt.figure(figsize=(9, 6))
    accuracies = tmp_log["accuracy_development"].values()
    generations = tmp_log["accuracy_development"].keys()
    plt.plot(generations, accuracies)
    # plt.title("Accuracy Development", fontsize=20)
    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Training Accuracy", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/accuracy_development")
    else:
        plt.show()
    plt.clf()


def fr4(ga, tmp_log):
    # Check how often, after recombination the child was better than its parents throughout the genealogy
    better_both_count = 0
    better_one_count = 0
    worse = 0
    for ind in ga.population.individuals:
        genealogy = get_genealogy_bfs(ind, ga.population.age)
        for individual in genealogy.keys():
            if len(genealogy.get(individual)) == 2:
                if individual.fitness > genealogy.get(individual)[0].fitness and \
                        individual.fitness > genealogy.get(individual)[1].fitness:
                    better_both_count += 1
                elif individual.fitness > genealogy.get(individual)[0].fitness or \
                        individual.fitness > genealogy.get(individual)[1].fitness:
                    better_one_count += 1
                else:
                    worse += 1
    print("better_both_count:", better_both_count)
    print("better_one_count:", better_one_count)
    print("worse:", worse)


def fr5(ga, tmp_log):
    for individual in ga.population.individuals:
        leaves = get_leaves_dfs(individual, ga.population.age)
        leaves_that_are_in_the_final_population_count = 0
        for leaf in leaves:
            if leaf in ga.population.individuals:
                leaves_that_are_in_the_final_population_count += 1
            print(leaf.origin, leaf.age)
        print(leaves_that_are_in_the_final_population_count)


def fr6(ga, tmp_log):
    better_both_counts = [0] + [a for (a, b, c) in tmp_log["goodness_of_children"].values()]
    better_one_counts = [0] + [b for (a, b, c) in tmp_log["goodness_of_children"].values()]
    worse_counts = [0] + [c for (a, b, c) in tmp_log["goodness_of_children"].values()]
    generations =  [0] + list(tmp_log["goodness_of_children"].keys())
    plt.figure(figsize=(9, 6))
    plt.bar(x=generations, height=worse_counts, width=0.8, align="edge", label="worse_count")
    plt.bar(x=generations, height=better_one_counts, width=0.8, bottom=np.array(worse_counts),
           align="edge", label="better_one_count")
    plt.bar(x=generations, height=better_both_counts, width=0.8,
           bottom=np.array(worse_counts)+np.array(better_one_counts), align="edge", label="better_both_counts")
    plt.xlabel("generation")
    plt.ylabel("children")
    plt.title("Recombine child-parents-goodness comparison")
    plt.legend()
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/recombine_child_parents_goodness_comparison")
    else:
        plt.show()
    plt.clf()


def fr7(ga, tmp_log):
    top_recombine_counts = [0] + list(tmp_log["top_recombine_counts"].values())
    top_mutate_counts = [0] + list(tmp_log["top_mutate_counts"].values())
    top_migrate_counts = [0] + list(tmp_log["top_migrate_counts"].values())
    generations = [0] + list(tmp_log["top_recombine_counts"].keys())
    plt.figure(figsize=(9, 6))
    # bar plot
    """plt.bar(x=generations, height=top_migrate_counts, width=0.3, align="edge",
           label="migration")
    plt.bar(x=generations, height=top_mutate_counts, width=0.3,
           bottom=np.array(top_migrate_counts), align="edge", label="mutation", color="gold")
    plt.bar(x=generations, height=top_recombine_counts, width=0.3,
           bottom=np.array(top_migrate_counts) + np.array(top_mutate_counts),
           align="edge", label="recombination", color="crimson")"""

    # stack plot
    plt.stackplot(generations, top_migrate_counts, top_mutate_counts, top_recombine_counts,
                  labels=["migration", "mutation", "recombination"], colors=["blue", "orange", "green"],
                  baseline="zero", alpha=0.6)

    plt.xlabel("generation", fontsize=16)
    plt.ylabel("top origins", fontsize=16)
    plt.title("Top 25% new individuals origin comparison", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/top_individuals_origin_comparison_stackplot")
    else:
        plt.show()
    plt.clf()


def fr8(ga, tmp_log):
    relative_diversities = tmp_log["relative_diversity_development"].values()
    generations = tmp_log["relative_diversity_development"].keys()
    plt.figure(figsize=(9, 6))
    plt.plot(generations, relative_diversities, linewidth=1.0)
    plt.title("relative diversity development", fontsize=20)
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("relative diversity", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/relative_diversity_development")
    else:
        plt.show()
    plt.clf()


def fr9(ga, tmp_log):
    """ to read e.g.: with open(f"{path}/lth/pickled_objects/pickled_final_population", 'rb') as pickle_file:
                          final_population = dill.load(pickle_file) """
    if tmp_log.get("log_path"):
        if not os.path.isdir(tmp_log["log_path"] + "/pickled_objects"):
            os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_final_population", "wb") as dill_file:
            dill.dump(ga.population, dill_file)


def fr10(ga, tmp_log):
    plt.figure(figsize=(9, 6))
    num_param = len(ga.population.individuals[0].genome)
    generations = tmp_log["max_0_count_of_highest_accuracy_development"].keys()
    plt.plot(generations, (np.array(list(tmp_log["max_0_count_of_highest_accuracy_development"].values()))/num_param)*100, label="Top Accuracy Individual")
    plt.plot(generations, (np.array(list(tmp_log["max_0_count_development"].values()))/num_param)*100, label="Highest Sparsity in Population")
    plt.plot(generations, (np.array(list(tmp_log["overall_max_0_count_development"].values()))/num_param)*100, label="Highest Sparsity in Evolution")
    # plt.title("Sparsity Development", fontsize=20)
    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Sparsity in %", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/sparsity_development")
    else:
        plt.show()
    plt.clf()


def fr11(ga, tmp_log):
    final_population_flattened = {}
    for rank, individual in enumerate(ga.population.individuals):
        final_population_flattened[rank] = {"genome": individual.genome, "origin": individual.origin, "fitness": individual.fitness}

    if tmp_log.get("log_path"):
        if not os.path.isdir(tmp_log["log_path"] + "/pickled_objects"):
            os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_final_population_flattened", "wb") as dill_file:
            dill.dump(final_population_flattened, dill_file)


def fr12(ga, tmp_log):
    plt.figure(figsize=(9, 6))
    generations = tmp_log["bound_development"].keys()
    plt.plot(generations, tmp_log["bound_development"].values())
    plt.title("accuracy bound development", fontsize=20)
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("accuracy bound", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/accuracy_bound_development")
    else:
        plt.show()
    plt.clf()


def deepcopy_population(population):
    population_copy = Population(population.size, population.mig_rate, population.mut_rate, population.rec_rate, population.par_rate, population.ga_config, population.individuals)
    population_copy.age = population.age
    return population_copy


def fr13(ga, tmp_log):
    # Evaluate final population on test dataset
    population = deepcopy_population(ga.population)
    population.ga_config.parents_evaluation.train_loader = ga.test_loader
    population.ga_config.individual_evaluation.train_loader = ga.test_loader
    population.ga_config.evaluation_cache = None
    population.ga_config.individual_evaluation.quality_metric = accuracy
    population.ga_config.parents_evaluation.performance_maximizing = True
    parents_evaluation = population.ga_config.parents_evaluation

    final_evaluated_population = parents_evaluation.evaluate_population(population, None)
    final_evaluated_population_flattened = {}
    for rank, individual in enumerate(final_evaluated_population.individuals):
        final_evaluated_population_flattened[rank] = {"genome": individual.genome, "origin": individual.origin,
                                                      "fitness": individual.fitness}
    if tmp_log.get("log_path"):
        with open(tmp_log.get("log_path") + "/pickled_objects/pickled_final_evaluated_population_flattened", "wb") as dill_file:
            dill.dump(final_evaluated_population_flattened, dill_file)
    print(final_evaluated_population)
    print("best test dataset accuracy:", final_evaluated_population.individuals[0].fitness)


def fr14(ga, tmp_log):
    dataset_instance = ga.args.dataset
    nn_architecture = get_NN_architecture_from_model(ga.model)
    if isinstance(dataset_instance, MOONS):
        subnet = get_subnetwork(ga.model, ga.population.individuals[0].genome, nn_architecture, ga.device)
        dataset_instance.plot_learned_function(False, subnet)


def fr15(ga, tmp_log):
    """ Increase sparsity maximally """
    population = deepcopy_population(ga.population)
    population.ga_config.parents_evaluation.train_loader = ga.test_loader
    population.ga_config.individual_evaluation.train_loader = ga.test_loader
    population.ga_config.evaluation_cache = None
    population.ga_config.individual_evaluation.quality_metric = accuracy
    population.ga_config.parents_evaluation.performance_maximizing = True
    individual_evaluation = population.ga_config.individual_evaluation
    parents_evaluation = population.ga_config.parents_evaluation

    if isinstance(population, dict):
        # For running routine from pickled files
        best_individual_genome = deepcopy(population[0]["genome"])
    else:
        # Use best performing individual on training dataset
        # best_individual_genome = deepcopy(population.individuals[0].genome)
        # Use best performing individual on testing dataset
        final_evaluated_population = parents_evaluation.evaluate_population(population, None)
        best_individual_genome = deepcopy(final_evaluated_population.individuals[0].genome)

    initial_quality = individual_evaluation.evaluate_individual(best_individual_genome, None)
    print("Initial Fitness:", initial_quality)

    num_param = len(best_individual_genome)
    initial_sparsity = (np.count_nonzero(best_individual_genome == 0)/num_param)*100
    removed_ones = 0
    performance_maximizing = parents_evaluation.performance_maximizing

    for i in range(len(best_individual_genome)):
        if best_individual_genome[i] == 1:
            best_individual_genome[i] = 0
            current_quality = individual_evaluation.evaluate_individual(best_individual_genome, None)

            # Only keep change if pruning does not reduce accuracy
            if performance_maximizing and current_quality < initial_quality:
                best_individual_genome[i] = 1
            elif not performance_maximizing and current_quality > initial_quality:
                best_individual_genome[i] = 1
            else:
                removed_ones += 1

    current_quality = individual_evaluation.evaluate_individual(best_individual_genome, None)
    current_sparsity = (np.count_nonzero(best_individual_genome == 0)/num_param)*100

    print("Maximally sparse best solution:", best_individual_genome)
    print("Fitness:", current_quality)
    print(f"Removed {removed_ones} 1s.")
    print(f"Increased sparsity from {initial_sparsity}% -> {current_sparsity}%")

    print(list(best_individual_genome))

    # Store in tmp_log
    tmp_log["max_pruned_SLT"] = {"bit_vector": best_individual_genome, "current_quality": current_quality,
                                 "sparsity_development": f"{initial_sparsity}% -> {current_sparsity}%",
                                 removed_ones: f"{removed_ones}"}

    # Save max_pruned_model
    max_pruned_subnet = ga.model.apply_mask(best_individual_genome)
    model_path = os.path.join(ga.logger.log_path, 'model_max_pruned.pt')
    torch.save(max_pruned_subnet.net.state_dict(), model_path)
    ga.best_model = max_pruned_subnet


def fr16(ga, tmp_log):
    """ This method has to be called last. """
    if tmp_log.get("log_path"):
        if not os.path.isdir(tmp_log["log_path"] + "/pickled_objects"):
            os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_tmp_log", "wb") as dill_file:
            dill.dump(tmp_log, dill_file)


final_routines_dict = {1: (fr1, [ir1, ir11], [er13]), 2: (fr2, [ir5], [er7]), 3: (fr3, [ir6], [er8]),
                       4: (fr4, [], []), 5: (fr5, [], []), 6: (fr6, [ir7], [er9]), 7: (fr7, [ir8], [er10]),
                       8: (fr8, [ir9], [er6]), 9: (fr9, [], []), 10: (fr10, [ir12], [er14]), 11: (fr11, [], []),
                       12: (fr12, [ir14], [er15]), 13: (fr13, [], []), 14: (fr14, [], []), 15: (fr15, [ir15], []),
                       16: (fr16, [], [])}