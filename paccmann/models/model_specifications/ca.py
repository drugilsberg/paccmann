"""Model specification based on the attention-only paradigm."""
import tensorflow as tf
from ...hyperparams import ACTIVATION_FN_FACTORY, POSITIONAL_ENCODING_FACTORY
from ...layers import (
    dense_attention_layer, embedding_layer, contextual_attention_layer
)


def ca_fn(features, labels, mode, params):
    """
    Implement model for IC50 prediction based on selected genes attention and
    smiles attention. Implements the contextual attention model.

    Args:
        - features: features for the observations (<dict<string,tf.Tensor>>).
        - labels: labels associated (<tf.Tensor>).
        - mode: mode for the model (<tf.estimator.ModeKeys>).
        - params: parameters for the model (<dict<string, object>>).
            Mandatory parameters are:
                - selected_genes_name: name of the selected genes features
                    (<string>).
                - tokens_name: name of the tokens features (<string>).
                - smiles_vocabulary_size: size of the tokens vocabulary
                    (<int>).
                - smiles_embedding_size: dimension of tokens' embedding
                    (<int>).
                - smiles_attention_size:  size of attentive units (<int>).
            Optional parameters for the model:
                - stacked_dense_hidden_sizes:  sizes of the hidden dense
                    layers (<list<int>>).
                - batch_norm: whether batch normalization applies at concatena-
                    tion of genes and smiles features, default: False (<bool>)
                - positional_encoding: whether positional encoding is applied.
                    Choose (<string>, None) and string from
                    {learned, sinusoidal}.
            Example params:
            ```
            {  
                "selected_genes_name": "selected_genes_10",
                "tokens_name": "smiles_atom_tokens",
                "smiles_vocabulary_size": 40,
                "smiles_embedding_size": 16,
                "positional_encoding": None,
                "smiles_attention_size": 256,
                "stacked_dense_hidden_sizes": [512, 256, 64, 16],
                "batch_norm": True
            }
            ```
    Returns:
        The predictions in the form of a 1D `tf.Tensor` and a prediction
        dictionary (<dict<string, tf.Tensor>>).   
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    # For de-standardization of IC50 prediction.
    min_ic50 = params.get('min', 0.0)
    max_ic50 = params.get('max', 0.0)

    # NOTE: dropout considered only during training.
    dropout = 0.0
    if is_training:
        dropout = params.get('dropout', 0.0)

    activation_fn = ACTIVATION_FN_FACTORY[params.get('activation', 'relu')]

    smiles_attention_size = params.get('smiles_attention_size', 256)

    genes = features[params['selected_genes_name']]
    tokens = features[params['tokens_name']]
    sequence_length = tokens.shape[1].value

    embedded_tokens = embedding_layer(
        tokens,
        params['smiles_vocabulary_size'],
        params['smiles_embedding_size'],
        name='smiles_embedding'
    )

    if params.get('positional_encoding', None) is not None:
        positional_encoding_fn = POSITIONAL_ENCODING_FACTORY[
            params['positional_encoding']
        ]
        positional_encoding = positional_encoding_fn(
            sequence_length, params['smiles_embedding_size'],
            name='positional_encoding'
        )
        # Implement additive positional encodings.
        embedded_tokens = tf.add(
                embedded_tokens, positional_encoding,
                name='additive_positional_embedding'
        )

    encoded_genes, gene_attention_coefficients = dense_attention_layer(
        genes, return_alphas=True,
        name='gene_attention'
    )

    encoded_smiles, smiles_attention_coefficients = contextual_attention_layer(
            genes, embedded_tokens, smiles_attention_size,
            return_alphas=True, name='smiles_attention'
    )

    features = tf.concat(
        [encoded_genes, encoded_smiles], axis=1,
        name='concatenated_genes_and_smiles'
    )

    # Apply batch normalization if specified.
    layer = (
        tf.layers.batch_normalization(features, training=is_training)
        if params.get('batch_norm', False) else features
    )

    for index, dense_hidden_size in enumerate(
        params.get('stacked_dense_hidden_sizes', [])
    ):
        layer = tf.layers.dropout(
            tf.layers.dense(
                layer, dense_hidden_size,
                activation=activation_fn,
                name='dense_hidden_{}'.format(index)
            ),
            rate=dropout,
            training=is_training,
            name='dropout_dense_hidden_{}'.format(index)
        )

    predictions = tf.squeeze(
        tf.layers.dense(layer, 1, name='logits'),
        axis=1,
        name='predictions'
    )

    prediction_dict = {
        'IC50': predictions,
        'IC50_micromolar': tf.exp(
            predictions * (max_ic50 - min_ic50) + min_ic50
        ),
        'smiles_attention': smiles_attention_coefficients,
        'gene_attention': gene_attention_coefficients,
        'features': features
    }

    return predictions, prediction_dict