# `chemprop` with uncertainty
This branch extends the message passing neural networks for molecular property prediction as described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237)
with uncertainty, as described in the paper [Evaluating Scalable Uncertainty Estimation Methods for DNN-Based Molecular Property Prediction](https://arxiv.org/abs/1910.03127).

This branch is currently under active development.

For uncertainty-specific instructions and differences with respect to the base model see Section [Uncertainty Estimation](#uncertainty-estimation).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Conda](#option-1-conda)
  * [Option 2: Docker](#option-2-docker)
  * [(Optional) Installing `chemprop` as a Package](#optional-installing-chemprop-as-a-package)
  * [Notes](#notes)
- [Web Interface](#web-interface)
- [Data](#data)
- [Training](#training)
  * [Train/Validation/Test Splits](#train-validation-test-splits)
  * [Cross validation](#cross-validation)
  * [Ensembling](#ensembling)
  * [Hyperparameter Optimization](#hyperparameter-optimization)
  * [Additional Features](#additional-features)
    * [RDKit 2D Features](#rdkit-2d-features)
    * [Custom Features](#custom-features)
- [Predicting](#predicting)
- [Uncertainty Estimation](#uncertainty-estimation)
- [TensorBoard](#tensorboard)
- [Results](#results)

## Requirements

While it is possible to run all of the code on a CPU-only machine, GPUs make training significantly faster. To run with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation

### Option 1: Conda

The easiest way to install the `chemprop` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chemprop`
3. `conda env create -f environment.yml`
4. `source activate chemprop` (or `conda activate chemprop` for newer versions of conda)
5. (Optional) `pip install git+https://github.com/bp-kelley/descriptastorus`

The optional `descriptastorus` package is only necessary if you plan to incorporate computed RDKit features into your model (see [Additional Features](#additional-features)). The addition of these features improves model performance on some datasets but is not necessary for the base model.

### Option 2: Docker

Docker provides a nice way to isolate the `chemprop` code and environment. To install and run our code in a Docker container, follow these steps:

1. Install Docker from [https://docs.docker.com/install/](https://docs.docker.com/install/)
2. `cd /path/to/chemprop`
3. `docker build -t chemprop .`
4. `docker run -it chemprop:latest /bin/bash`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs. 

### (Optional) Installing `chemprop` as a Package

If you would like to use functions or classes from `chemprop` in your own code, you can install `chemprop` as a pip package as follows:

1. `cd /path/to/chemprop`
2. `pip install -e .`

Then you can use `import chemprop` or `from chemprop import ...` in your other code.

### Notes

**PyTorch GPU:** Although PyTorch is installed automatically along with `chemprop`, you may need to install the GPU version manually. Instructions are available [here](https://pytorch.org/get-started/locally/).

**kyotocabinet**: If you get warning messages about `kyotocabinet` not being installed, it's safe to ignore them.
   
## Web Interface

For those less familiar with the command line, we also have a web interface which allows for basic training and predicting. After installing the dependencies following the instructions above, you can start the web interface in two ways:

1. Run `python web/run.py` and then navigate to [localhost:5000](http://localhost:5000) in a web browser. This will start the site in development mode.
2. Run `gunicorn --bind {host}:{port} 'wsgi:build_app()'`. This will start the site in production mode.
   * To run this server in the background, add the `--daemon` flag.
   * Arguments including `init_db` and `demo` can be passed with this pattern: `'wsgi:build_app(init_db=True, demo=True)'` 
   * Gunicorn documentation can be found [here](http://docs.gunicorn.org/en/stable/index.html).

![Training with our web interface](web/app/static/images/web_train.png "Training with our web interface")

![Predicting with our web interface](web/app/static/images/web_predict.png "Predicting with our web interface")


## Data

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks.

Our model can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
...
```
Datasets from [MoleculeNet](http://moleculenet.ai/) and a 450K subset of ChEMBL from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

## Training

To train a model, run:
```
python train.py --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is either "classification" or "regression" depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
python train.py --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
```

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

### Train/Validation/Test Splits

Our code supports several methods of splitting data into train, validation, and test sets.

**Random:** By default, the data will be split randomly into train, validation, and test sets.

**Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding `--split_type scaffold_balanced`.

**Separate val/test:** If you have separate data files you would like to use as the validation or test set, you can specify them with `--separate_val_path <val_path>` and/or `--separate_test_path <test_path>`.

Note: By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with `--split_sizes <train_frac> <val_frac> <test_frac>`. For example, the default setting is `--split_sizes 0.8 0.1 0.1`. Both also involve a random component and can be seeded with `--seed <seed>`. The default setting is `--seed 0`.

### Cross validation

k-fold cross-validation can be run by specifying `--num_folds <k>`. The default is `--num_folds 1`.

### Ensembling

To train an ensemble, specify the number of models in the ensemble with `--ensemble_size <n>`. The default is `--ensemble_size 1`.

### Hyperparameter Optimization

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the [hyperopt](https://github.com/hyperopt/hyperopt) package) in `hyperparameter_optimization.py`. This script finds the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:
```
python hyperparameter_optimization.py --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>
```
where `<n>` is the number of hyperparameter settings to try and `<config_path>` is the path to a `.json` file where the optimal hyperparameters will be saved. Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:
```
python train.py --data_path <data_path> --dataset_type <type> --config_path <config_path>
```

### Additional Features

While the model works very well on its own, especially after hyperparameter optimization, we have seen that adding computed molecule-level features can further improve performance on certain datasets. Features can be added to the model using the `--features_generator <generator>` flag.

#### RDKit 2D Features

As a starting point, we recommend using pre-normalized RDKit features by using the `--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the `--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of `rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling.

Note: In order to use the `rdkit_2d_normalized` features, you must have `descriptastorus` installed. If you installed via conda, you can install `descriptastorus` by running `pip install git+https://github.com/bp-kelley/descriptastorus`. If you installed via Docker, `descriptastorus` should already be installed.

The full list of available features for `--feagrtures_generator` is as follows. 

`morgan` is binary Morgan fingerprints, radius 2 and 2048 bits.
`morgan_count` is count-based Morgan, radius 2 and 2048 bits.
`rdkit_2d` is an unnormalized version of 200 assorted rdkit descriptors. Full list can be found at the bottom of our paper: https://arxiv.org/pdf/1904.01561.pdf
`rdkit_2d_normalized` is the CDF-normalized version of the 200 rdkit descriptors.

#### Custom Features

If you would like to load custom features, you can do so in two ways:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in `chemprop/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy `.npy` file or as a `.csv` file, you can load the features by using `--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that `.csv` files must have a header row and the features should be comma-separated with one line per molecule.
 
## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
python predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv
```
or
```
python predict.py --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv
```

## Uncertainty Estimation
This branch (`chemprop-uncertainty`) extends `chemprop` to output uncertainty estimates for each predicted property.
In particular, the aleatoric and the epistemic uncertainty for each prediction can be output.
See the paper [Evaluating Scalable Uncertainty Estimation Methods for DNN-Based Molecular Property Prediction](https://arxiv.org/abs/1910.03127)
for more details about the meaning of these two types of uncertainty and the theory behind their computation with the different methods.



Currently, using chemprop-uncertainty, for each predicted property three columns are output instead of one.
For each predicted property `x` the model outputs the columns: `x`, `x_ale_unc` and `x_epi_unc`. This holds even if no uncertainty is computed: in this case, `x_ale_unc` and `x_epi_unc` default to 0.

Currently, uncertainty estimation is supported only for regression (`--dataset_type regression`).

If no uncertainty-specific flag is provided, no uncertainty is estimated and the model behaves exactly as the base `chemprop` (with the only difference of the additional columns which default to 0).

In the following the uncertainty-specific flags to add uncertainty calculation using different methods are described.

### Aleatoric uncertainty

Aleatoric uncertainty estimation (distributional parameter estimation, Gaussian distribution) can be added to the model by specifying `--aleatoric` during training.

The  model trained with this additional parameter can be used to predict new molecules as usual. For each property `x`, the aleatoric uncertainty will be output in the `x_ale_unc` column.


### Epistemic uncertainty

#### Deep Ensembles

To estimate epistemic uncertainty using deep ensembles:
* Train the model as an ensemble (flag `--ensemble_size`, see [Ensembling](#ensembling))
* Predict providing the flag `--estimate_variance`.

For each property `x`, the epistemic uncertainty will be output in the `x_epi_unc` column.

#### MC-Dropout

To estimate epistemic uncertainty using MC-Dropout:
* Train the model with the additional flag `--epistemic mc_dropout`. This changes the model to include MC-Dropout with Concrete Dropout. 
* Predict specifying the `--sampling_size` flag. For example, `--sampling_size 20` corresponds to using 20 Monte Carlo samples.

For each property `x`, the epistemic uncertainty will be output in the `x_epi_unc` column.

Notice that in this implementation of MC-Dropout the dropout probability is not specified, since it is automatically learned using Concrete Dropout.
When training with `--epistemic mc_dropout`, the additional flag `--regularization_scale` is available. This sets the regularization scale for Concrete Dropout (default: 1e-4). See the [Concrete Dropout paper](https://arxiv.org/abs/1705.07832) for more details about its usage.


#### Bootstrapping

To estimate epistemic uncertainty using bootstrapping:
* Train the model as an ensemble (flag `--ensemble_size`, see [Ensembling](#ensembling)), specifying also the `--bootstrapping` flag.
* Predict providing the flag `--estimate_variance`.

For each property `x`, the epistemic uncertainty will be output in the `x_epi_unc` column.

### Compute uncertainty in practice

Aleatoric and epistemic uncertainty can be predicted together.

Example: aleatoric uncertainty + deep ensembles (to predict epistemic uncertainty):
```
python train.py --dataset_type regression ... --aleatoric --ensemble_size N
python predict.py ... --estimate_variance
```

Where `N` corresponds to the number of model's instances.

In this case, for each property `x`, both `x_ale_unc` and `x_epi_unc` take values.



## TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:6006](http://localhost:6006).
