import tensorflow as tf
from ...layers import dense_attention_layer, embedding_layer
from ...hyperparams import ACTIVATION_FN_FACTORY


def scnn_fn(features, labels, mode, params):
    """
    Implement model for IC50 prediction based on selected genes attention and
    attentive cnn.

    Args:
        - features: features for the observations (<dict<string,tf.Tensor>>).
        - labels: labels associated (<tf.Tensor>).
        - mode: mode for the model (<tf.estimator.ModeKeys>).
        - params: parameters for the model (<dict<string, object>>).
            Mandatory parameters are:
                - selected_genes_name: name of the selected genes features
                    (<string>).
                - tokens_name: name of the tokens features (<string>).
                - smiles_embedding_size: dimension of tokens' embedding
                    (<int>).
                - smiles_vocabulary_size: size of the tokens vocabulary
                    (<int>).

            Optional parameters for the model:
                - stacked_dense_hidden_sizes: sizes of the hidden dense
                    layers (<list<int>>).
                - smiles_attention_type: type of attention to be applied on
                    encoded smiles. <string> from {sequence, merged}.
                - smiles_attention: whether a sequence attention layer is
                    applied after embedding SMILES (before CNN) <bool>.
                - smiles_attention_size: size of the attentive layer for the
                    smiles sequence (<int>).
                - post_smiles_attention: whether attention should be applied
                    after cnn. Choose from {sequence, merged}, default False.
                - post_smiles_attention_size: size of post-cnn attention layer,
                    only applies in case post_smiles_attentions is not False.
                - post_smiles_reduction: whether time dimension of post-cnn
                    attention is reduced (<bool>). Defaults to True.
            NOTE:
                - The kernel sizes should match the dimensionality of the
                    smiles_embedding_size, so if the latter is 8, the
                    images are sequence_length x 8, then treat the 8 embedding
                    dimensions like channels in an RGB image.
                - For stacked_dense_hidden_sizes not more than 2 layers
                    should be required.

            Example params:
            ```
            {  
                "selected_genes_name": "selected_genes_10",
                "tokens_name": "smiles_atom_tokens",
                "smiles_vocabulary_size": 28, (28 for atoms, 30 for chars)
                "smiles_embedding_size": 8, (don't use more than 16)
                "filters": [128, 128],
                "kernel_sizes": [[10, 10], [20, 20]],
                "stacked_dense_hidden_sizes": [8, 4]
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

    dropout = 0.0
    if is_training:
        dropout = params.get('dropout', 0.0)

    genes = features[params['selected_genes_name']]
    tokens = features[params['tokens_name']]

    activation_fn = ACTIVATION_FN_FACTORY[params.get('activation', 'relu')]

    encoded_genes, gene_attention_coefficients = dense_attention_layer(
        genes, return_alphas=True, name='gene_attention'
    )

    # NOTE: tokens.shape[1].value = sequence_length = embedding_size.
    embedded_tokens = embedding_layer(
        tokens,
        params['smiles_vocabulary_size'],
        params['smiles_embedding_size'],
        name='smiles_embedding'
    )

    filters = params.get('filters', [128, 128, 128])
    kernel_sizes = params.get(
        'kernel_sizes',
        [
            [5, params['smiles_embedding_size']],
            [10, 1],
            [20, 1]
        ]
    )

    assert len(filters) == len(kernel_sizes)
    if len(filters) > 0:
        # NOTE: Treat the sequence embedding matrix as image.
        layer = tf.expand_dims(embedded_tokens, 3)
        for index, (filter_size, kernel_size) in enumerate(
            zip(filters, kernel_sizes)
        ):
            layer = tf.layers.conv2d(
                        inputs=layer, filters=filter_size,
                        kernel_size=kernel_size, padding='valid',
                        activation=activation_fn, name='conv_{}'.format(index)
            )
            # NOTE: Asymmetric stride is applied to only reduce sequence
            # length.
            layer = tf.layers.max_pooling2d(
                inputs=layer,
                pool_size=[2, 1],
                strides=[2, 1],
                name='pool_{}'.format(index)
            )

            encoded_smiles_cnn = tf.reshape(
                layer,
                [-1, layer.shape[1].value*layer.shape[2].value*filters[-1]]
            )

    layer = tf.layers.batch_normalization(
        tf.concat(
            [encoded_smiles_cnn, encoded_genes],
            axis=1,
        ),
        training=is_training,
        name='batch_normed_concatenated_genes_and_smiles'
    )

    # NOTE: stacking dense layers as a bottleneck.
    for index, dense_hidden_size in enumerate(
        params.get('stacked_dense_hidden_sizes', [])
    ):
        layer = tf.layers.dropout(
            activation_fn(
                tf.layers.batch_normalization(
                    tf.layers.dense(
                        layer, dense_hidden_size,
                        activation=None,
                        name='dense_hidden_{}'.format(index)
                    ),
                    training=is_training,
                    name='batch_normed_dense_{}'.format(index)
                ),
                name='ouputs_dense_{}'.format(index)
            ),
            rate=dropout,
            training=is_training,
            name='dropout_dense_hidden_{}'.format(index)
        )

    predictions = tf.squeeze(
        tf.layers.dense(layer, 1, name='logits'),
        axis=1, name='predictions'
    )

    prediction_dict = {
        'IC50': predictions,
        'IC50_micromolar': tf.exp(
            predictions * (max_ic50 - min_ic50) + min_ic50
        ),
        'gene_attention': gene_attention_coefficients
    }

    return predictions, prediction_dict