# Finding Strong Lottery Ticket Networks with Genetic Algorithms

---
by Philipp Altmann, Julian Schönberger, Maximilian Zorn, and Thomas Gabor

arxiv link: Will follow

This code contains the functionalities for the genetic algorithm and the basis for repeating 
the experiments from our paper "Finding Strong Lottery Ticket Networks with Genetic Algorithms".

## Code Overview

The ```src/``` directory contains the source code and is further divided into three subdirectories:
  - ```genetic_algorithm/```: Here, the functionality of the genetic algorithm is defined. 
    The file ```GA.py``` is the central control interface of the GA. In our studies we
    tested various means of generating individuals (cf. ```generation.py```), evaluating them
    (cf. ```evaluate_methods```), performing mutation (cf. ```mutation_methods```) and recombination
    (cf. ```recombination_methods```), plus selecting proper parents and the final survivors that make it 
    into the next generation (cf. ```selection_methods```). 
  - ```neural_network/```: This directory contains the base functions for the feedforward neural networks,
    as well as the implementation of subnetworks (cf. ```subnetwork.py```) and various parameter 
    initialization methods (cf. ```reinitialization_methods.py```).
  - ```experiments/base```: The base experiment class can be found in ```experiment_framework.py```. 
    An overview of the GA methods that are used by the GA configurations ("GA" and "GA (adaptive AB)") discussed in our paper
    (and some additional ones), can be found in ```GA_configurations.py```. A logging system allows information to be logged
    at different steps during the evolution (cf. ```init_routines.py```, ```evol_routines.py```). 
    This information is than used in ```final_routines.py``` to calculate and visualize some basic
    metrics like accuracy, sparsity or population diversity. 
  - ```experiments/GA_performance_comparison```: To rerun the experiments of ```Section 5``` select the proper experiment method
    in the main function. 
  - ```experiments/backprop_performance_comparison```: We included mean accuracies of networks trained with backpropagation 
    as a baseline in our experiments. ```hyperparameter_tuning``` contains the code for determining the best hyperparameter values. 
    To reenact the data gathering for backprop, the corresponding experiment methods can be easily selected in the main function
    (cf. ```Main.py```).
  - ```experiments/Edge_popup_performance_comparison```: This subdirectory contains the code for the 
    algorithm edge-popup by Ramanujan et al. (arxiv link: https://arxiv.org/abs/1911.13299), which is subject to the terms 
    of the ```Apache 2.0 License```. (For more information, see the license text in the `LICENSE` file).
    We mainly adapted the network generation, included our generation method and added code for 
    the different datasets. (The modifications are indicated in the respective files.) Since we don't use CUDA in our GA experiments we also disabled CUDA support for edge-popup. 
    The modified configurations that are used to run the code analogously to https://github.com/allenai/hidden-networks can be found in ```hidden-networks/configs/smallscale/fc/```.
  
The results of the GA and the backpropagation experiments are stored in ```resources/test_results/```. 
When executing edge-popup, the results are stored in a newly created runs folder.

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
