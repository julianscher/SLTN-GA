import copy
from src.genetic_algorithm.generation import generate_strict_connectivity, generate_lax_connectivity, \
    generate_lax_connectivity_alternative, generate_lax_connectivity_learned_accuracy_filter, \
    generate_negligent
from src.genetic_algorithm.mutation_methods import single_point_mutation_path_fixed, single_point_mutation, \
    path_flip_mutation, single_point_mutation_negligent_optimized
from src.genetic_algorithm.recombination_methods import crossover_recombination_path_fixed, crossover_recombination, \
    neuron_recombination, crossover_recombination_negligent
from src.genetic_algorithm.selection_methods import select_by_cutoff, select_by_graceful_cutoff, better_select_by_walk

##### Configs used for experiments #####

# top_speed is "GA"
top_speed = {"generate_method": generate_negligent, "mutate_method": single_point_mutation_negligent_optimized,
             "recombine_method": crossover_recombination_negligent, "selection_method": select_by_cutoff,
             "selection_args": {}, "generate_args": {}, "mutate_args": {"connectivity_scheme": "negligent"},
             "recombine_args": {"connectivity_scheme": "negligent"}}

# max_acc_bound is "GA (adaptive AB)"
max_acc_bound = {"generate_method": generate_lax_connectivity_learned_accuracy_filter, "mutate_method": single_point_mutation,
                 "recombine_method": crossover_recombination, "selection_method": select_by_cutoff,
                 "selection_args": {}, "generate_args": {"bound": 0.85, "max_reas_gen_time": 4, "max_giv_gen_time": 5},
                 "mutate_args": {"connectivity_scheme": "lax"}, "recombine_args": {"connectivity_scheme": "lax"}}


##### Other configs #####

# max_acc is "GA (static AB)"
max_acc = {"generate_method": generate_lax_connectivity_alternative, "mutate_method": single_point_mutation,
           "recombine_method": crossover_recombination, "selection_method": select_by_cutoff,
           "selection_args": {}, "generate_args": {"bound": 0.7}, "mutate_args": {"connectivity_scheme": "lax"},
           "recombine_args": {"connectivity_scheme": "lax"}}

max_acc_bound_small = copy.deepcopy(max_acc_bound)
max_acc_bound_small["generate_args"] = {"bound": 0.7, "max_reas_gen_time": 4, "max_giv_gen_time": 5}

top_spars_and_acc = {"generate_method": generate_strict_connectivity, "mutate_method": single_point_mutation,
                     "recombine_method": crossover_recombination, "selection_method": select_by_graceful_cutoff,
                     "selection_args": {}, "generate_args": {"input_output_paths": None}, "mutate_args": {"connectivity_scheme": "strict"},
                     "recombine_args": {"connectivity_scheme": "lax"}}

top_avg_acc = {"generate_method": generate_lax_connectivity_alternative, "mutate_method": single_point_mutation_path_fixed,
               "recombine_method": neuron_recombination, "selection_method": select_by_graceful_cutoff,
               "selection_args": {}, "generate_args": {"bound": 0.7}, "mutate_args": {"connectivity_scheme": "lax", 'alpha': 0.5},
               "recombine_args": {"connectivity_scheme": "soft"}}

top_speed_old = {"generate_method": generate_strict_connectivity, "mutate_method": single_point_mutation,
                 "recombine_method": crossover_recombination, "selection_method": select_by_cutoff,
                 "selection_args": {}, "generate_args": {"input_output_paths": None}, "mutate_args": {"connectivity_scheme": "lax"},
                 "recombine_args": {"connectivity_scheme": "lax"}}

ref_point = {"generate_method": generate_lax_connectivity, "mutate_method": path_flip_mutation,
             "recombine_method": crossover_recombination_path_fixed, "selection_method": better_select_by_walk,
             "selection_args": {}, "generate_args": {}, "mutate_args": {"connectivity_scheme": "lax"},
             "recombine_args": {"connectivity_scheme": "lax"}}