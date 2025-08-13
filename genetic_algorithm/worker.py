import os
import sys
import time

from args import args
from genetic_algorithm.ga import GeneticAlgorithm
from genetic_algorithm.ga_configuration.ga_config_factory import GAConfigFactory
from genetic_algorithm.logging.logger import Logger
from utilities.helper_functions import get_results_path, set_seed
from utilities.parser import get_model, load_config, get_dataset, get_criterion, get_routines, \
    get_test_metric, get_population_config, get_ga_config_dict, get_termination_condition

sys.setrecursionlimit(10000)

# This class should have a very similar interface as the backprop worker for a single model
class Worker:
    def __init__(self):
        self._init()
        model = get_model(args.model, args)

        if args.device != "cpu":
            print(f"Use GPU-{args.device} for training")
            model.to(args.device)
        else:
            print(f"Use CPU for training")

        # For Multi-task learning, create a worker (call the main) for each new task and load pretrained model

        self.model = model
        self.device = args.device
        self.data = get_dataset()
        self.criterion = get_criterion()
        self.logger = Logger(get_routines(), args.log_every, self.log_path if args.log else None)
        self.ga_config = GAConfigFactory().create_ga_config(get_ga_config_dict(), self)

    def run(self):
        train_dataloader = self.data.train_loader
        val_dataloader = None
        test_dataloader = self.data.test_loader
        ga = GeneticAlgorithm(self.model, self.criterion, train_loader=train_dataloader, val_loader=val_dataloader,
                              test_loader=test_dataloader, device=args.device, logger=self.logger,
                              population_config=get_population_config(),
                              termination_condition=get_termination_condition(), ga_config=self.ga_config, args=args)

        self.logger.apply_init_routines(ga)

        # TODO: Log start state
        start_time = time.time()
        ga.evolve()
        evolution_time = time.time() - start_time

        self.logger.apply_final_routines(ga)

        start_time = time.time()
        test_metric = get_test_metric()
        total_loss, total_metric = ga.test_top_individual(test_metric)
        print(total_loss, total_metric)
        #self.data.plot_learned_function(train_dataset=False, subnet=ga.best_model, log_path=self.log_path)
        testing_time = time.time() - start_time

        # TODO: Log end state

        ga.save_model()

    def _init(self):
        load_config()
        log_path = f"{get_results_path(args.results_path)}/{args.name}"
        print(args)

        # Log settings
        if args.log:
            path_components = log_path.split("/")
            partial_path = ""
            for path_component in path_components:
                partial_path += path_component + "/"
                if not os.path.isdir(partial_path):
                    os.mkdir(partial_path)

            # Differentiate between jobs started from within an experiment and runs directly invoked by the worker
            if not "job" in log_path.split("/")[-1]:
                runs = os.listdir(log_path)
                run_numbers = [int(run[3:]) for run in runs if run[:3] == "run"]
                next_run_number = max(run_numbers) + 1 if run_numbers else 0
                log_path += f"/run{next_run_number}"
                os.mkdir(log_path)
            with open(log_path + "/settings.txt", "w") as text_file:
                print(args, file=text_file)

        self.log_path = log_path
        set_seed(args.seed, args.only_model)

if __name__ == "__main__":
    worker = Worker()
    worker.run()