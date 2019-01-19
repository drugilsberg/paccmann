"""Model specification based on selected genes and hand-engineered drug features."""
import tensorflow as tf
from ...hyperparams import ACTIVATION_FN_FACTORY
from ...layers import dense_attention_layer


def dnn_fn(features, labels, mode, params):
    """
    Implement model for IC50 prediction based on selected genes and
    hand-engineered drug features.

    Args:
        - features: features for the observations (<dict<string,tf.Tensor>>).
        - labels: labels associated (<tf.Tensor>).
        - mode: mode for the model (<tf.estimator.ModeKeys>).
        - params: parameters for the model (<dict<string, object>>).
            Mandatory parameters are:
                - selected_genes_name: name of the selected genes features
                    (<string>).
                - drug_features_name: name of the drug features (<string>).
                - stacked_dense_hidden_sizes:  sizes of the hidden dense 
                    layers (<list<int>>).
            Optional parameters are:
                - batch_norm: Whether batch normalization applies at concatena-
                    tion of genes and smiles features, default: False (<bool>)
                - genes_attention: Whether a dense attention layer should be 
                    applied on the genes data, default: False (<bool>)
                - drugs_attention: Whether a dense attention layer should be 
                    applied on the fingerprints data, default: False (<bool>)
            Example params:
            ```
            {  
                "selected_genes_name": "selected_genes_10",
                "drug_features_name": "fingerprints_512",
                "stacked_dense_hidden_sizes": [8, 8, 4],
                "drugs_attention": False,
                "genes_attention": False,
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

    genes = features[params['selected_genes_name']]
    drug_features = tf.cast(
        features[params['drug_features_name']],
        tf.float32
    )

    # Optionally use dense attention on the inputs.
    genes = (
        dense_attention_layer(
            genes, return_alphas=False,
            name='gene_attention'
        ) if params.get('genes_attention', False) else
        genes
    )
    drug_features = (
        dense_attention_layer(
            drug_features, return_alphas=False,
            name='drugs_attention'
        ) if params.get('drugs_attention', False) else
        drug_features
    )

    layer = tf.concat(
        [genes, drug_features], axis=1,
        name='concatenated_genes_and_drug_features'
    )

    # Apply batch normalization if specified.
    layer = (
        tf.layers.batch_normalization(layer, training=is_training)
        if params.get('batch_norm', False) else layer
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
        axis=1, name='predictions')

    prediction_dict = {
        'IC50': predictions,
        'IC50_micromolar': tf.exp(
            predictions * (max_ic50 - min_ic50) + min_ic50
        ),
    }

    return predictions, prediction_dict