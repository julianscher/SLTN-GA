import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    "--init", default="default", help="The method used for network parameter initialization"
)
parser.add_argument(
    "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Used device for tensor and matrix operations (default: cpu)"
)
parser.add_argument(
    "--models", default=None,
    help="When you want to define multiple models"
        + "architecture: Used neural network architecture"
        + "model: Used neural network model type"
        + "activation: Used activation function"
        + "prune_rate: The initial prune rate for the model (default: 0.5)"
        + "init: The method used for network parameter initialization"
        + "pretrained: The method used for network parameter initialization"
        + "criterion: The loss function to use (default: CE)"
)
parser.add_argument(
    "--model", default="MLP", help="Used neural network type (default: MLP)"
)
parser.add_argument(
    "--architecture", default=[2, 100, 2], help="Used neural network architecture."
)
parser.add_argument(
    "-a", "--activation", default="relu", help="Used activation function."
)
parser.add_argument(
    "-pr", "--prune-rate", default=0.5, type=float, help="The initial prune rate for the model (default: 0.5)"
)
parser.add_argument(
    "--pretrained", default=None, type=str, help="Pretrained model path"
)
parser.add_argument(
    "--results-path", default="../study_out", help="Where to store the run results (absolute path). "
                                         "If None, study_out will be used."
)
parser.add_argument(
    "--optimizer", default="Adam", type=str, help="The neural network optimizer to use"
)
parser.add_argument(
    "-lr", "--learning-rate", default=1e-3, type=float, help="Learning rate (default: 1e-3)"
)
parser.add_argument(
    "-wd", "--weight-decay", default=1e-4, type=float, help="Weight decay (default: 1e-4)"
)
parser.add_argument(
    "--criterion", default="CE", type=str, help="The loss function to use (default: CE)"
)
parser.add_argument(
    "--n-epochs", default=100, type=int, help="Number of epochs to train (default: 100)"
)
parser.add_argument(
    "--dataset", default="MOONS", type=str, help="The dataset to use (default: CIRCLES)"
)
parser.add_argument(
    "--normalized", default=False, type=bool, help="Whether or not to normalize the data (default: False)"
)
parser.add_argument(
    "--batch_size", default=16, type=int, help="The batch size for training (default: 128)"
)
parser.add_argument(
    "--dataset-args", default={}, help="Any additional dataset arguments"
)
parser.add_argument(
    "--test-metric", default="Accuracy", type=str, help="The metric to evaluate on the test set"
)
parser.add_argument(
    "--only_inference", default=False, help="Whether the training should be skipped and the model should only be evaluated"
)
parser.add_argument(
    "--config", default="configs/ga/examples/top_speed.yml", help="The config file used to configure the run"
)
parser.add_argument(
    "--name", default="run" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    help="The identifier for the run results folder."
)
parser.add_argument(
    "--seed", default=69, help="The seed for making the runs deterministic."
)
parser.add_argument(
    "--only-model", default=False, help="If seeding should only fix the model."
)
parser.add_argument(
    "--log", default=True, help="If information about the run should be logged and results should be stored"
)
parser.add_argument(
    "--mask_except", default=["net.0.bias", "net.2.bias"], help="Which model parameters should not be included in the mask"
)

args = parser.parse_args()
