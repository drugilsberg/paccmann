[![Build Status](https://github.com/drugilsberg/paccmann/actions/workflows/build.yml/badge.svg)](https://github.com/drugilsberg/paccmann/actions/workflows/build.yml)

# DISCLAIMER:
This code gives the `tensorflow` implementation of PaccMann as of our paper in [Molecular Pharmaceutics](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520).
We recommend to use the `pytorch` implementation available in the [paccmann_predictor](https://github.com/PaccMann/paccmann_predictor) package.

# PaccMann

`paccmann` is a package for drug sensitivity prediction and is the core component of the repo.

The package provides a toolbox of learning models for IC50 prediction using drug's chemical properties and tissue-specific cell lines gene expression.

### Citation
Please cite us as follows:
``` bib
@article{manica2019paccmann,
  title={Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders},
  author={Manica, Matteo and Oskooei, Ali and Born, Jannis and Subramanian, Vigneshwari and S{\'a}ez-Rodr{\'\i}guez, Julio and Mart{\'\i}nez, Mar{\'\i}a Rodr{\'\i}guez},
  journal={Molecular pharmaceutics},
  volume={16},
  number={12},
  pages={4797--4806},
  year={2019},
  publisher={ACS Publications},
  doi = {10.1021/acs.molpharmaceut.9b00520},
  note = {PMID: 31618586}
}
```

## Installation

### Setup of the virtual environment

We strongly recommend to work inside a virtual environment (`venv`).

Create the environment:

```sh
python3 -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

### Module installation

The module can be installed either in editable mode:

```sh
pip3 install -e .
```

Or as a normal package:

```sh
pip3 install .
```

## Models training

Models can be trained using the script `bin/training_paccmann` that is installed together with the module. Check the [examples](https://github.com/drugilsberg/paccmann/tree/master/examples) for a quick start.
For more details see the help of the training command by typing `training_paccmann -h`:

```console
usage: training_paccmann [-h] [-save_checkpoints_steps 300]
                         [-eval_throttle_secs 60] [-model_suffix]
                         [-train_steps 10000] [-batch_size 64]
                         [-learning_rate 0.001] [-dropout 0.5]
                         [-buffer_size 20000] [-number_of_threads 1]
                         [-prefetch_buffer_size 6]
                         train_filepath eval_filepath model_path
                         model_specification_fn_name params_filepath
                         feature_names

Run training of a `paccmann` model.

positional arguments:
  train_filepath        Path to train data.
  eval_filepath         Path to eval data.
  model_path            Path where the model is stored.
  model_specification_fn_name
                        Model specification function. Pick one of the
                        following: ['dnn', 'rnn', 'scnn', 'sa', 'ca', 'mca'].
  params_filepath       Path to model params. Dictionary with parameters
                        defining the model.
  feature_names         Comma separated feature names. Select from the
                        following: ['smiles_character_tokens',
                        'smiles_atom_tokens', 'fingerprints_256',
                        'fingerprints_512', 'targets_10', 'targets_20',
                        'targets_50', 'selected_genes_10',
                        'selected_genes_20', 'cnv_min', 'cnv_max', 'disrupt',
                        'zigosity', 'ic50', 'ic50_labels'].

optional arguments:
  -h, --help            show this help message and exit
  -save_checkpoints_steps 300, --save-checkpoints-steps 300
                        Steps before saving a checkpoint.
  -eval_throttle_secs 60, --eval-throttle-secs 60
                        Throttle seconds between evaluations.
  -model_suffix , --model-suffix 
                        Suffix for the trained moedel.
  -train_steps 10000, --train-steps 10000
                        Number of training steps.
  -batch_size 64, --batch-size 64
                        Batch size.
  -learning_rate 0.001, --learning-rate 0.001
                        Learning rate.
  -dropout 0.5, --dropout 0.5
                        Dropout to be applied to set and dense layers.
  -buffer_size 20000, --buffer-size 20000
                        Buffer size for data shuffling.
  -number_of_threads 1, --number-of-threads 1
                        Number of threads to be used in data processing.
  -prefetch_buffer_size 6, --prefetch-buffer-size 6
                        Prefetch buffer size to allow pipelining.
```
