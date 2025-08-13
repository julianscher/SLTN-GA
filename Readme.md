# Towards Scalable Lottery Ticket Networks using Genetic Algorithms 

---
by Julian Schönberger, Maximilian Zorn, Jonas Nüßlein, Thomas Gabor, and Philipp Altmann

arXiv link: https://arxiv.org/abs/2508.08877

This code contains the functionalities for the genetic algorithm and the basis for repeating 
the experiments from our paper *"Towards Scalable Lottery Ticket Networks using Genetic Algorithms"*.
This is an extended version of the original algorithm from the paper *"Finding Strong Lottery Ticket Networks with Genetic Algorithms"*.

The original implementation from the first paper is preserved and tagged as ```v1.0.0``` in this repository.
Use ```v1.0.0``` to replicate the original *"Finding Strong Lottery Ticket Networks with Genetic Algorithms"* experiments.
Use ```v2.0.0``` for the updated *"Towards Scalable Lottery Ticket Networks using Genetic Algorithms"* experiments and all new features.

---

The repository includes:

* **Genetic Algorithm implementation** tailored for evolving sparse neural network topologies,
  supporting configurable generation, mutation, recombination, and selection strategies.
* **Experiment runner framework** for reproducing the results from our paper *"Towards Scalable Lottery Ticket Networks using Genetic Algorithms"*
  as well as the earlier work *"Finding Strong Lottery Ticket Networks with Genetic Algorithms"*.
* **Config-driven design** — all experiment parameters, datasets, and GA settings are specified via YAML files for reproducibility and easy modification.
* **Extensibility** — the codebase is structured to support new datasets, architectures, and GA operators with minimal changes.

This extended version enhances the original algorithm with:

* Improved modularity and clearer separation between GA operators and experiment orchestration.
* Support for larger-scale experiments via efficient logging, checkpointing, and parallel task handling.
* Additional datasets and experiment configurations.
* Updated Worker interface for seamless integration with YAML-based experiment definitions.
* Code refactoring for clarity, maintainability, and adaptability to future research directions.

---

## Code Overview

The `src/` directory contains the source code and is organized into the following subdirectories:

* `configs/`: YAML files defining datasets, GA operators (and their parameters), hyperparameters, observer routines, and neural-network architecture setups used to reproduce and extend the paper’s experiments.
* `data/`: Code to generate the *moons*, *circles*, *digits*, and *blobs* datasets used in the paper.
* `genetic_algorithm/`: Core genetic-algorithm implementation.

  * `ga_configuration/`: Modular code to configure and initialize a GA instance from methods, arguments, and hyperparameters declared in YAML.
  * `generation/`, `evaluation/`, `mutation/`, `recombination/`, `selection/`: Operator implementations for candidate generation, evaluation, mutation, crossover, and parent/survivor selection.
  * `worker.py`: Central controller orchestrating GA evolution.
  * `population.py`, `individuals.py`: Data structures for populations and individuals.
  * `logging/`: Observer routines for tracking/logging during evolution and general final plotting routines invoked per run.
* `models/`: Components for feed-forward neural networks (see `fc/`), a general subnetwork wrapper with masking (`subnetwork/`), and various parameter-initialization methods (`reinitialization_methods.py`).
* `utilities/`: Helper functions, including bit-vector utilities, subnetwork visualization, profiling tools, and a general backpropagation trainer/worker.
* `experiments/GA_performance_comparison/`: To reproduce the experiments from **Section 5**, select the appropriate experiment method in the main entry point.
* `experiments/backprop_performance_comparison/`: Baseline experiments trained with backpropagation.

  * `hyperparameter_tuning/` contains the code used to determine optimal hyperparameters.
  * To recreate the backpropagation baselines, select the corresponding experiment methods in `Main.py`.
* `experiments/Edge_popup_performance_comparison/`: Code for the edge-popup algorithm by Ramanujan et al. (arXiv: 1911.13299), used under the **Apache 2.0 License** (see `LICENSE`). We adapted network generation, added our generation method and dataset support, and disabled CUDA to match our GA setup. Modified configurations mirroring `allenai/hidden-networks` are in `hidden-networks/configs/smallscale/fc/`.

**Results:** GA and backpropagation outputs are stored in `study_out/`. Edge-popup runs are written to a newly created `runs/` directory.


For further details on how to work with edge-popup, we refer to https://github.com/allenai/hidden-networks.

## Setup
1. Set up a virtualenv with python 3.10.11.
2. Run ```pip install -r requirements.txt``` to meet the requirements
3. For the edge-popup experiments you can use the same python version, but you have to create 
   a different venv and install the requirements from ```src/experiments/Edge_popup_performance_comparison/hidden-networks/requirements.txt```

## Requirements (GA)
```
contourpy==1.2.0
cycler==0.12.1
dill==0.3.7
filelock==3.13.1
fonttools==4.47.2
fsspec==2023.12.2
Jinja2==3.1.3
joblib==1.3.2
kiwisolver==1.4.5
line-profiler==4.1.2
MarkupSafe==2.1.3
matplotlib==3.8.2
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
packaging==23.2
pandas==2.1.4
pillow==10.2.0
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
scikit-learn==1.3.1
scipy==1.11.4
six==1.16.0
sympy==1.12
threadpoolctl==3.2.0
torch==2.1.2
typing_extensions==4.9.0
tzdata==2023.4
xxhash==3.4.1
accelerate==1.4.0
torchvision==0.16.0
```

## Requirements (edge-popup)
```
absl-py==2.0.0
cachetools==5.3.2
certifi==2023.11.17
charset-normalizer==3.3.2
filelock==3.13.1
fsspec==2023.12.2
google-auth==2.26.2
google-auth-oauthlib==1.2.0
grpcio==1.60.0
idna==3.6
Jinja2==3.1.3
joblib==1.3.2
Markdown==3.5.2
MarkupSafe==2.1.3
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
oauthlib==3.2.2
packaging==23.2
pillow==10.2.0
protobuf==4.23.4
pyasn1==0.5.1
pyasn1-modules==0.3.0
PyYAML==6.0.1
requests==2.31.0
requests-oauthlib==1.3.1
rsa==4.9
scikit-learn==1.3.2
scipy==1.11.4
six==1.16.0
sympy==1.12
tensorboard==2.15.1
tensorboard-data-server==0.7.2
threadpoolctl==3.2.0
torch==2.1.2
torchvision==0.16.2
tqdm==4.66.1
typing_extensions==4.9.0
urllib3==2.1.0
Werkzeug==3.0.1
```
