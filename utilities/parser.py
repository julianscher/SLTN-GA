import os
import sys
from types import SimpleNamespace

import yaml
import torch
from torch import nn

import data
import genetic_algorithm.logging.termination_conditions as ga_termination_conditions
import genetic_algorithm.logging.init_routines as init_routines
import genetic_algorithm.logging.evol_routines as evol_routines
import genetic_algorithm.logging.final_routines as final_routines
from args import args
from models.fc.mlp import MLP
from models.fc.mlp_embedded import MLPEmbedded
from models.subnetworks.base_subnetwork import BaseSubnetwork
from models.subnetworks.subnetworks import SubnetMLP, SubnetMLPEmbedded
from utilities.helper_functions import get_project_root_path
from utilities.net_utils import accuracy, cross_entropy_loss


def load_config():
    # Use config args if defined
    if args.config:
        print(f"Loading arguments from {args.config}")
        if not os.path.isfile(args.config):
            config_file = open(str(get_project_root_path()) + "/" + str(args.config)).read()
        else:
            config_file = open(args.config)
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        none_keys = {key: None for key, value in config.items() if value == "None"}
        config.update(none_keys)

        # Overwrite args
        args.__dict__.update(config)

    # Overwrite with existing command line arguments
    com_args = {}
    for idx, k in enumerate(sys.argv):
        if idx != 0 and idx%2 != 0:
            if k.startswith("--"):
                com_args[k[2:]] = sys.argv[idx + 1]
            elif k.startswith("-"):
                com_args[k[1:]] = sys.argv[idx + 1]

    # Overwrite args
    args.__dict__.update(com_args)


def get_ga_config_dict():
    ga_config_dict = {"generation": {"method": args.generation_method, "args": args.generation_args},
                      "selection": {"method": args.survivor_selection_method, "args": args.survivor_selection_args},
                      "mutation": {"method": args.mutation_method, "args": args.mutation_args},
                      "recombination": {"method": args.recombination_method, "args": args.recombination_args},
                      "survivor_evaluation": {"method": args.survivor_evaluation_method, "args": args.survivor_evaluation_args},
                      "parents_evaluation": {"method": args.parents_evaluation_method, "args": args.parents_evaluation_args},
                      "individual_evaluation": {"method": args.individual_evaluation_method, "args": args.individual_evaluation_args},
                      "evaluation_cache": {"type": args.evaluation_cache_type, "args": args.evaluation_cache_args}}
    return ga_config_dict

def get_model(model_type, args):
    print('==> Building model..')
    if model_type == "MLP":
        model = SubnetMLP(args.architecture, args.init, args.prune_rate, args.activation, args.mask_except, args.device)
        if args.pretrained:
            load_pretrained(model, args)
    elif model_type == "MLPEmbedded":
        model = MLP(args.trained_model_architecture, args.init, args.prune_rate, args.activation, args.device)
        load_pretrained(model, args)
        model = SubnetMLPEmbedded(args.architecture, args.init, args.prune_rate, model, args.swap_ratio, args.activation,
                                  args.device)
    print("Model is running on:", next(model.net.parameters()).device)
    return model


def get_routines():
    init_rout = args.init_routines
    evol_rout = args.evol_routines
    final_rout = args.final_routines

    routines = {"init": [], "evol": [], "final": []}
    for r in init_rout:
        routines["init"].append(getattr(init_routines, r))

    for r in evol_rout:
        routines["evol"].append(getattr(evol_routines, r))

    for r in final_rout:
        routines["final"].append(getattr(final_routines, r))

    return routines


def get_termination_condition():
    method = getattr(ga_termination_conditions, args.termination_condition)
    termination_condition_args = args.termination_condition_args
    return SimpleNamespace(**{"method": method, "args": termination_condition_args})


def parse_take_only(input):
    if isinstance(input, str):
        try:
            # Safely evaluate the input string
            result = eval(input)
            # Check if the result is a list
            if isinstance(result, list) or isinstance(result, tuple):
                return result
            else:
                raise ValueError("The input does not evaluate to a list.")
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")
    return input


def get_dataset():
    if args.dataset == "BLOBS":
        dataset = getattr(data, args.dataset)(args.dataset_args["centers"], args.normalized, args.device)
    elif args.dataset == "DIGITS":
        dataset = getattr(data, args.dataset)(args.normalized, args.device, args.dataset_args["classes"], args.dataset_args["with_validation"])
    elif args.dataset == "MNIST":
        dataset = getattr(data, args.dataset)(args.normalized, args.device, args.dataset_args["classes"])
    else:
        dataset = getattr(data, args.dataset)(args.normalized, args.device)

    return dataset


def get_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return optimizer


def load_pretrained(model, args):
    if os.path.isfile(args.pretrained):
        print(f"Loading pretrained model from '{args.pretrained}'")
        pretrained = torch.load(args.pretrained)
        if isinstance(model, BaseSubnetwork):
            model_state_dict = model.net.state_dict()
            model_state_dict.update(pretrained)
            model.net.load_state_dict(model_state_dict)
        else:
            model_state_dict = model.state_dict()
            model_state_dict.update(pretrained)
            model.load_state_dict(model_state_dict)


def get_criterion():
    if args.criterion == "MSE":
        return nn.MSELoss().to(args.device)
    elif args.criterion == "MAE":
        return nn.L1Loss().to(args.device)
    elif args.criterion == "BCE":
        return nn.BCELoss().to(args.device)
    elif args.criterion == "CE":
        return cross_entropy_loss


def get_test_metric():
    if args.test_metric == "Accuracy":
        return accuracy
    elif args.test_metric == "MSE":
        return nn.MSELoss().to(args.device)


def get_quality_metric(quality_metric_name):
    if quality_metric_name == "Accuracy":
        return accuracy
    elif quality_metric_name == "MSE":
        return nn.MSELoss().to(args.device)
    elif quality_metric_name == "MAE":
        return nn.L1Loss().to(args.device)
    elif quality_metric_name == "CE":
        return cross_entropy_loss

def get_population_config():
    population_config = {"pop_size":args.pop_size, "mig_rate": args.mig_rate, "mut_rate": args.mut_rate, "rec_rate": args.rec_rate, "par_rate": args.par_rate}
    return population_config
