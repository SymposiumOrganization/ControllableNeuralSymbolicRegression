# Controllable Neural Symbolic Regression
This repository contains the code and resources for the paper "Controllable Neural Symbolic Regression" by Tommaso Bendinelli, Luca Biggio, and Pierre-Alexandre Kamienny.

### Overview
Neural Symbolic Regression with Hypotheses (NSRwH) is a novel neural symbolic regression method that allows the incorporation of user-defined prior knowledge or hypotheses about the expected structure of the ground-truth expression. The method improves both the accuracy and controllability of the predicted expression structure.


## Getting Started

### Prerequisites
* Tested on Python 3.9.5 
* Tested on Ubuntu 20.04
* Tested on PyTorch 1.12 
* Tested on Pytorch Lightning 1.9.5 (Does not work out of the box with Python Lightning >= 2.0.0)
### Installation
Clone the repository:
``` 
git clone https://github.com/SymposiumOrganization/ControllableNeuralSymbolicRegression.git
```
Create and source the virtual environment:
```
python3 -m venv env
source env/bin/activate
```
Install PyTorch from https://pytorch.org/get-started, version 1.12.0 is greatly recommended.
Install the ControllableNesymres package and its dependencies:
```
cd src/
pip install -e .
```
Download the weights of the trained model and place them in the model folder:
```
cd model/
```

Please check the requirements file if you encounter trouble with some other dependencies.

## Interactive Demo
### Hosted on HuggingFace
Try it out here: https://huggingface.co/spaces/TommasoBendinelli/ControllableNeuralSymbolicRegression

### Hosted locally
1. Download the weights from HuggingFace:
```
git clone   https://huggingface.co/TommasoBendinelli/ControllableNeuralSymbolicRegressionWeights 
```
2. Run the interactive demo using the demo module:
```
streamlit run visualization/demo.py
```


## Reproducing the Experiments
### Data Generation (Training)
Generate synthetic datasets using the data_generation module. For our experiments we used 200 million equations with the following parameters:
``` 
python3 scripts/data_creation/create_dataset.py  --root_folder_path target_folder  --number_of_equations 200000000 --cores 32 --resume
Arguments:
    --root_folder_path: path to the folder where the dataset will be saved
    --number_of_equations: number of equations to generate
    --cores: number of cores to use for parallelization
Additional Arguments:
    --resumed: if True, the dataset generation will be resumed from the last saved file
    --eq_per_block: number of equations to generate in each block
``` 
Note that:
1. With 65 cores it took more than 10 days to generate the 200M dataset. We recommend to use a smaller number of equations for testing purposes (e.g., 10M). 
2. In some cases the data generation process could hang. In this case, you can kill the process and resume the generation from the last saved file using the --resume flag.
3. Equations are generated randomly by a seed dependent on the current date and time, so you will get different equations every time you run the script. If you want to generate the same equations, you can set the seed manually in the script (line 754 of src/ControllableNesymres/dataset/generator.py)

Optionally, after generating the dataset you can remove equations that are numerically meaningless (i.e. for all x in the range, the equation is always 0, infinity, or NaN) using the following scripts:
1. Identify the equations to remove. This script will create a npy containing for equation a tuple (equation_idx, True/False) where False means that the equation is invalid. This file is called equations_validity.npy and is saved in the same folder as the dataset.
```
python3 scripts/data_creation/check_equations.py --data_path target_folder/the_dataset_folder
Arguments:
    --data_path: Path to the dataset created with create_dataset.py
Additional Arguments:
    --debug/--no-debug: if --no-debug, the script is run with multiprocessing
```
2. Create a new dataset with only the good equations. This script will create a new dataset in the same folder as the original one, but inside
a folder called "datasets" instead of "raw_datasets".
```
python3 scripts/data_creation/remove_invalid_equations.py --data_path target_folder/the_dataset_folder
```
Arguments:
    --data_path: Path to the dataset created with create_dataset.py
Additional Arguments:
    --debug/--no-debug: if --no-debug, the script is run with multiprocessing

### Benchmark Set handling 
If you want to reproduce the experiments on the benchmark set, you will need to convert from the csv format into the dataloader format. To do so, run the following script:
```
scripts/data_creation/convert_csv_to_dataload_format.py 
```
This script will create a new folder called "benchmark" inside the data folder. Inside this folder, it will create a folder for each benchmark set.


### Model Training
Train the NSRwH model using the model module:
``` 
python scripts/train.py train_path=target_folder/datasets/2000000 benchmark_path=data/validation
``` 
Note we make use of [Hydra](https://hydra.cc) to manage the configuration. The associated configuration file is located in scripts/config.py. You can change the configuration by either editing the file or by passing the desired parameters as command line arguments. For example, to train the model with a different number of epochs you can run:
```
python scripts/train.py  train_path=target_folder/datasets/2000000 benchmark_path=target_folder/datasets/2000 batch_size=100
```
Take a look at the configuration file for more details about the available parameters. The conditioning setction is located under dataset.
If you want to train the model without the conditioning, i.e. the standard NSR model, you can run:
```
python scripts/train.py  train_path=target_folder/datasets/2000000 benchmark_path=target_folder/datasets/2000 batch_size=100 dataset.conditioning.mode=False architecture.conditioning=False
```

Note that by default the model will test on the benchmark dataset every check_val_every_n_epoch epochs. Please note that if you have not created the benchmark dataset, you will neet to avoid validation by setting check_val_every_n_epoch to a very large number (e.g., 1000000) and saving the model according to the steps.

## Citation

If you use this code or our results in your research, please cite the following paper:
``` 
@article{bendinelli2023controllable,
  title={Controllable Neural Symbolic Regression},
  author={Bendinelli, Tommaso and Biggio, Luca and Kamienny, Pierre-Alexandre},
  journal={arXiv preprint arXiv:2304.10336},
  year={2023}
}
``` 


## License
This project is licensed under the MIT License - see the LICENSE file for details.
