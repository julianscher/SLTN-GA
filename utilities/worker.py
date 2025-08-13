import os
import time

import torch

from args import args
from utilities.helper_functions import get_results_path, set_seed
from utilities.parser import load_config, get_dataset, get_optimizer, get_criterion, get_test_metric, \
    load_pretrained, get_model
from utilities.trainer import Trainer


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
        self.optimizer = get_optimizer(model, args)
        self.data = get_dataset()
        self.criterion = get_criterion()

    def run(self, lr_scheduler=None, logger=None):
        train_dataloader = self.data.train_loader
        test_dataloader = self.data.test_loader
        trainer = Trainer(self.model, self.optimizer, self.criterion, train_dataloader, test_loader=test_dataloader,
                          device=args.device, PATH=self.log_path)

        # TODO: Log start state
        if not args.only_inference:
            start_time = time.time()
            trainer.train(args.n_epochs)
            training_time = time.time() - start_time

        start_time = time.time()
        test_metric = get_test_metric()
        total_loss, total_metric = trainer.test(test_metric)
        print(total_loss, total_metric.item() if isinstance(total_metric, torch.Tensor) else total_metric)
        self.data.plot_learned_function(train_dataset=False, subnet=self.model, log_path=self.log_path)
        testing_time = time.time() - start_time

        # TODO: Log end state

        trainer.save_model()

    def _init(self):
        load_config()
        log_path = f"{get_results_path(args.results_path)}/{args.name}"

        # Log settings
        if args.log:
            path_components = log_path.split("/")
            partial_path = ""
            for path_component in path_components:
                partial_path += path_component + "/"
                if not os.path.isdir(partial_path):
                    os.mkdir(partial_path)
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
    # Note: You have to remove the "config" value in args.py to run this!
    worker = Worker()
    worker.run()

