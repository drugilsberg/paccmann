import tensorflow as tf
from ...layers import (
    dense_attention_layer, embedding_layer,
    sequence_attention_layer, contextual_attention_layer,
    contextual_attention_matrix_layer
)
from ...hyperparams import ACTIVATION_FN_FACTORY
from ...datasets import assemble_cnv_gep_data


def mca_fn(features, labels, mode, params):
    """
    Implement model for IC50 prediction based on selected genes attention and
    a multiscale attentive cnn.

    Args:
        - features: features for the observations (<dict<string, tf.Tensor>>).
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
                - filters: numbers of filters to learn per convolutional 
                    layer (<list<int>>).
                - kernel_sizes: xizes of kernels per convolutional layer
                    (<list<list<int>>>).
                - multiheads: amount of attentive multiheads per SMILES
                    embedding. (<list<int>>). Should have len(filters)+1
                - stacked_dense_hidden_sizes:  sizes of the hidden dense
                    layers (<list<int>>).
                - smiles_attention: type of attention to be applied on encoded
                    smiles. Default: None. <string> in
                    {"sequence", "contextual", "matrix"}.
                - smiles_attention_size: size of the attentive layer for the
                    smiles sequence (<int>).
                - smiles_reduction: whether time dimension of post-cnn
                    attention is reduced (<bool>). Defaults to True. 
                    Does not apply for matrix attention.
                NOTE: The kernel sizes should match the dimensionality of the
                            smiles_embedding_size, so if the latter is 8, the
                            images are sequence_length x 8, then treat the 8
                            embedding dimensions like channels in an RGB image.

            Example params:
            ```
            {  
                "selected_genes_name": "selected_genes_10",
                "tokens_name": "smiles_atom_tokens",
                "smiles_attention":  true,
                "smiles_attention_size": 8,
                "smiles_vocabulary_size": 28,
                "smiles_embedding_size": 8,
                "filters": [128, 128],
                "kernel_sizes": [[3, 8], [5, 8]],
                "multiheads":[32, 32, 32]
                "stacked_dense_hidden_sizes": [512, 64, 16]
            }
            ```
    Returns:
        The predictions in the form of a 1D `tf.Tensor` and a prediction
        dictionary (<dict<string, tf.Tensor>>).
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # For de-standardization of the IC50 prediction.
    min_ic50 = params.get('min', 0.0)
    max_ic50 = params.get('max', 0.0)

    dropout = params.get('dropout', 0.0) if is_training else 0.0
    batch_size = (
        params['batch_size']
        if is_training else params['eval_batch_size']
    )

    tokens = features[params['tokens_name']]
    sequence_length = tokens.shape[1].value

    # Use transcriptomics and genomics
    if (
        params.get('use_cnv_data', False) and
        params.get('use_gep_data', True)
    ):
        # Genes will be of shape
        # `[batch_size, num_cnv_features + gep (5), num_genes (2128)]`.
        genes = assemble_cnv_gep_data(
           features, features[params['selected_genes_name']]
        )
    # Use only transcriptomics.
    elif params.get('use_gep_data', True):
        genes = features[params['selected_genes_name']] 
    # Use only genomics.
    elif params.get('use_cnv_data', False):
        genes = assemble_cnv_gep_data(features)

    num_gene_features = 1 if len(genes.shape) == 2 else genes.shape[2].value

    activation_fn = ACTIVATION_FN_FACTORY[params.get('activation', 'relu')]

    def attention_list_to_matrix(coding_tuple, axis=2):
        """ 
        Unpack the attention weights.

        Args:
            - coding_tuple: a list of tuples (outputs, att_weights) 
                coming from the attention function.
            - axis: the dimension along which expansion takes place 
                to concatenate the attention weights.
        
        Returns:
            - raw_coeff: a `tf.Tensor` with the attention weights of all 
                multiheads and convolutional kernel sizes concatenated
                along last dimension.
            - coeff: a `tf.Tensor` with the attention weights averaged
                along the given axis.
        """
        raw_coeff = tf.concat(
            [tf.expand_dims(t[1], 2) for t in coding_tuple], axis=axis
        )
        coeff =  tf.reduce_mean(raw_coeff, axis=axis)
        return raw_coeff, coeff


    # NOTE: tokens.shape[1].value = sequence_length = embedding_size.
    embedded_tokens = embedding_layer(
        tokens, params['smiles_vocabulary_size'],
        params['smiles_embedding_size'],
        name='smiles_embedding'
    )

    filters = params.get('filters', [32, 32])
    kernel_sizes = params.get(
        'kernel_sizes',
        [
            [3, params['smiles_embedding_size']],
            [5, params['smiles_embedding_size']]
        ]
    )
    multiheads = params.get('multiheads', [16, 16, 16])
    assert len(filters) == len(kernel_sizes)
    assert len(filters)+1 == len(multiheads)

    if params.get('dense_attention', False) == False:
        # If no dense attention is applied on genes, the same, unfiltered
        # genes are given as context to every contextual layer.
        encoded_genes = [genes]*len(multiheads)
        gene_attention_coefficients = tf.zeros(
            [batch_size, genes.shape[1].value]
        )
    elif params.get('gene_multihead', False) == False:
        # Dense attention is applied, but only ones, i.e. the same context.
        encoded_genes, gene_attention_coefficients = (
            dense_attention_layer(
                genes, return_alphas=True, name='gene_attention'
            )
        )
        encoded_genes = [encoded_genes]*len(multiheads)
    elif params.get('gene_multihead', False):
        # Filter genes differently for each SMILES kernel size.
        gene_tuple = [
            dense_attention_layer(
                genes, return_alphas=True, 
                name='gene_attention_{}'.format(l)
            ) for l in range(len(multiheads))
        ]
        encoded_genes = [tpl[0] for tpl in gene_tuple]
        gene_attention_coefficients_multi, gene_attention_coefficients = (
            attention_list_to_matrix(gene_tuple, axis=2)
        )

    # NOTE: Treat the sequence embedding matrix as an image.
    # Apply batch norm after activation function.
    def pad_sequence(data, kernel_size):
        """ 
        Pad the sequence.

        Args:
            - data: a `tf.Tensor` of shape .
            - axis: The dimension along which expansion takes place 
                to concatenate the attention weights.
        
        Returns:
            - raw_coeff: a `tf.Tensor` with the attention weights of all 
                multiheads and convolutional kernel sizes concatenated
                along last dimension.
            - coeff: a `tf.Tensor` with the attention weights averaged
                along the given axis.
        """
        pad = tf.expand_dims(
            embedding_layer(
                tf.zeros([batch_size, 1], dtype=tf.int32),
                params['smiles_vocabulary_size'],
                params['smiles_embedding_size']
            ), axis=3, name='smiles_padding'
        )
        pad_size = kernel_size[0] // 2
        return tf.concat([pad]*pad_size + [data] + [pad]*pad_size, axis=1)

    inputs = tf.expand_dims(embedded_tokens, 3)
    # i-th element has shape `[batch_size, T, filters(i)]`.
    convolved_smiles = [
        tf.layers.batch_normalization(
            tf.layers.dropout(
                tf.squeeze(
                    tf.layers.conv2d(
                        inputs=pad_sequence(inputs, kernel_size),
                        filters=num_kernel, kernel_size=kernel_size,
                        padding='valid', activation=activation_fn,
                        name='conv_{}'.format(index)
                    ),  axis=2
                ), rate=dropout
            ), training=is_training
        ) for index, (num_kernel, kernel_size) in enumerate(
            zip(filters, kernel_sizes)
        )
    ]
    # Complement convolved smiles with residual connection.
    convolved_smiles.insert(0, embedded_tokens)
    
    # Attention mechanism.
    if params.get('smiles_attention', None) == 'sequence':
        encoding_coefficient_tuple = [
            sequence_attention_layer(
                convolved_smiles[layer],
                params.get('smiles_attention_size', 256), return_alphas=True,
                reduce_sequence=params.get('smiles_reduction', True),
                name='sequence_attention_{}'.format(layer)
            ) for layer in range(len(convolved_smiles))
            for ind in range(multiheads[layer])
        ]
    elif params.get('smiles_attention', None) == 'contextual':
        encoding_coefficient_tuple = [
            contextual_attention_layer(
                encoded_genes[layer], convolved_smiles[layer],
                params.get('smiles_attention_size', 256), return_alphas=True,
                reduce_sequence=params.get('smiles_reduction', True),
                name='contextual_attention_{}'.format(layer)
            ) for layer in range(len(convolved_smiles))
            for _ in range(multiheads[layer])
        ]
    elif params.get('smiles_attention', None) == 'matrix':
        encoding_coefficient_tuple = [
            contextual_attention_matrix_layer(
                genes, convolved_smiles[layer], return_scores=True
            ) for layer in range(len(convolved_smiles))
            for _ in range(multiheads[layer])
        ]
    elif params.get('smiles_attention', None) is not None:
        raise RuntimeError(
            'Unknown attention mechanism specified. Choose from '
            "{'sequence', 'contextual', 'matrix', None}."
        )

    # Done with attention, now prepare for concatenation with genes.
    # Check need to unpack list of tuples into encoded_smiles +
    # attention weights.
    if params.get('smiles_attention', None) is not None :
        if params.get('smiles_attention', None) == 'matrix':
            # Deal with attention weights first.
            # Each list entry of the tuple is of shape
            # `[batch_size, num_gene_features, sequence_length]`.
            attention_coefficients_raw, attention_coefficients = (
                attention_list_to_matrix(
                    encoding_coefficient_tuple, axis=3
                )
            )
            
            # Each output is shaped
            # `[batch_size, smiles_embedding_size, num_gene_features]`.
            encoded_smiles_list = [t[0] for t in encoding_coefficient_tuple]
            encoded_smiles = tf.concat(
                encoded_smiles_list, axis=1, name='encoded_smiles'
            )
            encoded_smiles.set_shape([
                batch_size, 
                (params['smiles_embedding_size']+num_gene_features) * 
                multiheads[0]+sum(
                    [
                        a*(b+num_gene_features)
                        for a, b in zip(multiheads[1:], filters)
                    ]
                )            
            ])
        # Applies for sequence or contextual attention
        else: 
            # Each alpha of the list of tuples is of shape
            # `[batch_size, sequence_length]`.
            # a_c_raw are of shape `[batch_size, T, multiheads * len(filters)]`
            # attention_coefficients is simply of shape `[batch_size, T]`.
            attention_coefficients_raw, attention_coefficients = (
                attention_list_to_matrix(
                    encoding_coefficient_tuple, axis=2
                )
            )
            encoded_smiles_list = [t[0] for t in encoding_coefficient_tuple]
            if params.get('smiles_reduction', True):
                # encoded_smiles is list of Tensors shape
                # `[batch_size, attention_size]`.
                encoded_smiles = tf.concat(
                        encoded_smiles_list, axis=1, name='encoded_smiles'
                )
                encoded_smiles.set_shape([
                    batch_size,
                    params['smiles_embedding_size']*multiheads[0] + 
                        sum([a * b for a, b in zip(multiheads[1:], filters)])
                ])

            else:
                # encoded_smiles is list of 3D Tensors of shape
                # `[batch_size, sequence_length, attention_size]`.
                encoded_smiles = [
                    tf.reshape(
                        encoded_smiles_list[layer],
                        [-1, sequence_length*filters[layer-1]]
                    ) for layer in range(1, len(encoded_smiles_list))
                ]
                encoded_smiles.insert(0, tf.reshape(
                    encoded_smiles_list[0],
                    [-1, sequence_length*params['smiles_embedding_size']]
                    )
                )
                encoded_smiles = tf.concat(
                    encoded_smiles, axis=1, name='encoded_smiles'
                )
                encoded_smiles.set_shape([
                    batch_size,
                    sequence_length * (
                        params['smiles_embedding_size']*multiheads[0]+ 
                        sum([a * b for a, b in zip(multiheads[1:], filters)])
                    )
                ])
    # In case no attention was applied
    else:
        encoded_smiles = [
            tf.reshape(
                convolved_smiles[layer+1],
                [-1, sequence_length*filters[layer]]
            ) for layer in range(len(convolved_smiles)-1)
        ]
        encoded_smiles.insert(0, tf.reshape(
                convolved_smiles[0],
                [-1, sequence_length*params['smiles_embedding_size']]
            )
        )

        encoded_smiles = tf.concat(
            encoded_smiles, axis=1, name='encoded_smiles'
        )

    # Apply batch normalization if specified
    layer = (
        tf.layers.batch_normalization(encoded_smiles, training=is_training)
        if params.get('batch_norm', False) else encoded_smiles
    )

    # NOTE: stacking dense layers as a bottleneck
    for index, dense_hidden_size in enumerate(
        params.get('stacked_dense_hidden_sizes', [])
    ):
        if not params.get('batch_norm', False):
            layer = tf.layers.dropout(
                tf.layers.dense(
                    layer, dense_hidden_size, activation=activation_fn,
                    name='dense_hidden_{}'.format(index)
                ),
                rate=dropout, training=is_training,
                name='dropout_dense_hidden_{}'.format(index)
            )
        # If batch_norm = True, look at position argument
        elif params.get('batch_norm_bef', True):
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
                rate=dropout, training=is_training,
                name='dropout_dense_hidden_{}'.format(index)
            )
        # Then, batch_norm is applied after activation
        else:
            layer = tf.layers.dropout(
                tf.layers.batch_normalization(
                    tf.layers.dense(
                        layer, dense_hidden_size, activation=activation_fn,
                        name='outputs_dense_{}'.format(index)
                    ),
                    training=is_training,
                    name='batch_normed_dense_{}'.format(index)
                ), rate=dropout, training=is_training,
                name='dropout_dense_hidden_{}'.format(index)
            )

    predictions = tf.squeeze(tf.layers.dense(
        layer, 1, name='logits'
    ))
    prediction_dict = {
        'gene_attention': gene_attention_coefficients,
        'smiles_attention': attention_coefficients,
        'smiles_attention_raw': attention_coefficients_raw,
        'features': encoded_smiles
    }       
    # Converts IC50 to micromolar concentration if scaling
    # parameters available.
    # If unavailable, concentration will default to exp(0)=1.
    prediction_dict.update({
        'IC50': predictions,
        'IC50_micromolar': tf.exp(
            predictions * (max_ic50 - min_ic50) + min_ic50
        ),
    })
    if params.get('gene_multihead', False):
        prediction_dict.update(
            {'gene_attention_raw': gene_attention_coefficients_multi}
        )

    return predictions, prediction_dict