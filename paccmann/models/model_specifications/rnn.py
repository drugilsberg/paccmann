"""Model specification based on selected genes attention and attentive rnn."""
import tensorflow as tf
from ...layers import (
    dense_attention_layer, embedding_layer,
    sequence_attention_layer, contextual_attention_layer
)
from ...hyperparams import RNN_CELL_FACTORY, ACTIVATION_FN_FACTORY


def rnn_fn(features, labels, mode, params):
    """
    Implement model for IC50 prediction based on selected genes attention and
    attentive rnn.

    Args:
        - features: features for the observations (<dict<string, tf.Tensor>>).
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
                - smiles_cell_size: size of the stacked rnn cells
                    (<list<int>>).

            Optional parameters for the model:
                - stacked_dense_hidden_sizes:  sizes of the hidden dense
                    layers (<list<int>>).
                - rnn_cell_type: type of RNN cell (<string>) from {lstm, gru}.
                    defaults to lstm units.
                - batch_norm: whether batch normalization applies at
                    concatenation of genes and smiles features.
                    Default: False (<bool>).
                - smiles_attention: whether attention is applied after the RNN
                    encoding. Default None, choose (<string>) from 
                    {sequence, context}.
                - smiles_attention_size: size of the attentive layer for the
                    smiles sequence (<int>).
                - smiles_reduction: whether time dimension of post-cnn
                    attention is reduced (<bool>). Defaults to True.

            Example params:
            ```
            {
                "selected_genes_name": "selected_genes_10",
                "tokens_name": "smiles_atom_tokens",
                "smiles_vocabulary_size": 40,
                "smiles_embedding_size": 16,
                "smiles_cell_size": [16, 8],
                "stacked_dense_hidden_sizes": [8, 4],
                "rnn_cell_type": "gru"
                "batch_norm": False
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

    genes = features[params['selected_genes_name']]
    tokens = features[params['tokens_name']]

    embedded_tokens = embedding_layer(
        tokens, params['smiles_vocabulary_size'],
        params['smiles_embedding_size'], name='smiles_embedding'
    )

    # NOTE:
    # - RNN cell type is user customizable.
    # - layers can be stacked, dropout on the output of every layer
    # (on the output since we use that one for attention).
    rnn_cell_fn = RNN_CELL_FACTORY[params.get('rnn_cell_type', 'gru')]

    cells_fw = [
        tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell_fn(layer_size),
            output_keep_prob=1 - dropout
        )
        for layer_size in params['smiles_cell_size']
    ]

    cells_bw = [
        tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell_fn(layer_size),
            output_keep_prob=1 - dropout
        )
        for layer_size in params['smiles_cell_size']
    ]

    output_steps, output_fw, output_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw, cells_bw=cells_bw,
        inputs=embedded_tokens, dtype=tf.float32,
        scope='smiles_embedding_encoding',
        # NOTE: added parallel iterations option because of this issue:
        # https://github.com/tensorflow/tensorflow/issues/19568.
        parallel_iterations=156
    )

    if params.get('smiles_attention', None) == 'sequence':
        encoded_smiles, smiles_attention = sequence_attention_layer(
            output_steps, params.get('smiles_attention_size', 256),
            return_alphas=True, reduce_sequence=True,
            name='smiles_attention'
        )
    elif params.get('smiles_attention', None) == 'contextual':
        encoded_smiles, smiles_attention = contextual_attention_layer(
            genes, output_steps, params.get('smiles_attention_size', 256),
            return_alphas=True, reduce_sequence=True,
            name='smiles_attention'
        )
    else:
        last_layer = len(params['smiles_cell_size']) - 1
        encoded_smiles = (
            tf.concat(
                [output_fw[last_layer].h, output_bw[last_layer].h], axis=1,
                name='concatenated_hidden_states'
            )
            if params.get('rnn_cell_type', 'gru') == 'lstm' else
            tf.concat(
                [output_fw[last_layer], output_bw[last_layer]], axis=1,
                name='concatenated_hidden_states'
            )
        )
        smiles_attention = tf.zeros([1])

    # NOTE: here we have encoded_smiles.shape =
    # `[batch_size, params['smiles_cell_size']]`.
    encoded_genes, gene_attention_coefficients = dense_attention_layer(
        genes, return_alphas=True,
        name='gene_attention'
    )

    layer = tf.concat(
        [encoded_genes, encoded_smiles], axis=1,
        name='concatenated_genes_and_smiles'
    )

    # Apply batch normalization if specified.
    layer = (
        tf.layers.batch_normalization(layer, training=is_training)
        if params.get('batch_norm', False) else layer
    )

    # NOTE: stacking dense layers as a bottleneck.
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
        'IC50_micromolar': tf.exp(predictions*(max_ic50-min_ic50)+min_ic50),
        'gene_attention': gene_attention_coefficients,
        'smiles_attention': smiles_attention
    }

    return predictions, prediction_dict