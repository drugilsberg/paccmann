# Examples

## Model parameters

Optimal model parameters for the training of the models implemented in the toolbox are stored in `model_params`.
The tuning has been performed on a POWER8 cluster, hence the models might be too large for training on a laptop.
The parameter files are in `json` format and each corresponds to a different encoder:

- Contextual attention encoder (`ca.json`)
- Dense encoder (`dnn.json`)
- Multiscale convolutional attention encoder (`mca.json`)
- Recurrent encoder (`rnn.json`)
- Sequence attention encoder (`sa.json`)
- Convolutional encoder (`scnn.json`)

## Toy data

In `data/train` and `data/test` we include a small collection of TFRecords compatible with the format used by the toolbox.

## Run a example

After installation, to run a training on the toy examples use `training_paccmann` passing the data, the desired encoder and the corresponding model parameters:

```sh
training_paccmann data/train data/test /path/to/store/model/ mca model_params/mca.json smiles_atom_tokens,selected_genes_20
```

For details on the TFRecords schema check [paccmann/datasets.py](https://github.com/drugilsberg/paccmann/blob/master/paccmann/datasets.py).
