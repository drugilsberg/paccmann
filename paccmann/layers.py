"""Custom layers implementation."""
import tensorflow as tf
from tensor2tensor.layers import common_attention


def sequence_attention_layer(
        inputs, attention_size, time_major=False,
        reduce_sequence=True, return_alphas=False, name=None
):
    """
    Code adapted from this
    [repo](https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py)
    Attention mechanism layer which reduces RNN/Bi-RNN outputs or sequence data
    with an attention vector.
    The idea was proposed in the article by Z. Yang et al.,
    "Hierarchical Attention Networks for Document Classification", 2016:
    http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article.

    Args:
        - inputs: the attention inputs.
            Matches 3D sequences data or RNN/Bi-RNN layer (not final state):
                In case of sequence data, this must be a `tf.Tensor`:
                    If time_major == False (default), this must be a tensor of
                    shape:
                        `[batch_size, sequence_length, hidden_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[sequence_length, batch_size, hidden_size]`.
                In case of RNN, this must be RNN outputs `tf.Tensor`:
                    If time_major == False (default), this must be a tensor of
                    shape:
                        `[batch_size, sequence_length, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[sequence_length, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw,
                outputs_bw) containing the forward and
                the backward RNN outputs `tf.Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `tf.Tensor` shaped:
                        `[batch_size, sequence_length, cell_fw.output_size]`
                        and outputs_bw is a `tf.Tensor` shaped:
                        `[batch_size, sequence_length, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `tf.Tensor` shaped:
                        `[sequence_length, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `tf.Tensor` shaped:
                        `[sequence_length, batch_size, cell_bw.output_size]`.
        - attention_size: linear size of the attention weights.
        - time_major: the shape format of the `inputs` Tensors.
            If true, these `tf.Tensors` must be shaped `[sequence_length,
            batch_size, depth]`.
            If false, these `tf.Tensors` must be shaped `[batch_size,
            sequence_length, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.
            However, most TensorFlow data is batch-major, so by default this
            function accepts input and emits output in batch-major form.
        - reduce_sequence: specifies whether, after filtering with attention
            weights, the average of the sequence is computed or not.
            If True, result is `[batch_size, hidden_size]`, else it is
            `[batch_size, sequence_length, hidden_size]`.
        - return_alphas: whether to return attention coefficients variable
            along with layer's output. Used for visualization purpose.
    Returns:
        In case of a 3D sequence, this will be a `tf.Tensor` shaped:
            `[batch_size, hidden_size]` or
            `[batch_size, sequence_length, hidden_size]`.
        In case of RNN, this will be a `tf.Tensor` shaped:
            `[batch_size, cell.output_size]` or
            `[batch_size, sequence_length, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `tf.Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]` or 
            `[batch_size, sequence_length, cell_fw.output_size + cell_bw.output_size]`.
    """
    with tf.variable_scope(
        name, default_name="sequence_attention_layer", values=[inputs]
    ):
        if isinstance(inputs, tuple):
            # in case of Bi-RNN, concatenate the forward and the backward RNN
            # outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # from `[sequence_length, batch_size, hidden_size]` to
            # `[batch_size, sequence_length, hidden_size]`
            inputs = tf.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value

        # Trainable parameters
        w_omega = tf.Variable(
            tf.random_normal([hidden_size, attention_size], stddev=0.1)
        )
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation
            # to each of the batch_size*sequence_length.
            # Shape of `v` is `[batch_size, sequence_length, attention_size]`
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size attention_size
        # from `v` is reduced with `u` vector
        # [batch_size, sequence_length]
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        # [batch_size, sequence_length]
        alphas = tf.nn.softmax(vu, name='alphas')

        # If reduce_sequence is true, result is `[batch_size, hidden_size]`
        # else it is `[batch_size, sequence_length, hidden_size]`
        output = (
            tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
            if reduce_sequence else
            inputs * tf.expand_dims(alphas, -1)
        )

        # Optionally return the attention weights
        return (
            (output, alphas)
            if return_alphas else
            output
        )


def dense_attention_layer(inputs, return_alphas=False, name=None):
    """
    Attention mechanism layer for dense inputs.

    Args:
        - inputs: attention inputs. This must be a `tf.Tensor` of shape:
        `[batch_size, feature_size]` or
        `[batch_size, feature_size, hidden_size]`.
        - return_alphas: whether to return attention coefficients variable
          along with layer's output. Used for visualization purpose.
    Returns:
        If return_alphas == False (default) this will be a `tf.Tensor` with
        shape: `[batch_size, feature_size]` else it will be a tuple
        (outputs, alphas) with the alphas being of shape
        `[batch_size, feature_size]`.
    """
    with tf.variable_scope(
        name, default_name="dense_attention_layer", values=[inputs]
    ):  
        # If input comes with a hidden dimension (e.g. 5 features per gene)
        if len(inputs.shape) == 3:
            inputs = tf.squeeze(
                tf.layers.dense(
                    inputs, 1, activation=tf.nn.relu, name='feature_collapse'
                ),
                axis=2
            )
        assert len(inputs.shape)==2
        feature_size = inputs.shape[1].value
        alphas = tf.layers.dense(
            inputs, feature_size,
            activation=tf.nn.softmax,
            name='attention'
        )
        output = tf.multiply(inputs, alphas, name='filtered_with_attention')

        return (
            (output, alphas)
            if return_alphas else
            output
        )


def embedding_layer(inputs, vocab_size, embed_size, name=None):
    """
    Implements an embedding layer

    Args:
        - inputs: attention inputs. This must be a `tf.Tensor` of type int
            and shape: `[batch_size, input_sequence_length]`.
        - vocab size: The size of the input token dictionary (int)
        - hidden size: The dimensionality of the embedding vectors (int).
    Returns:
        This will be a `tf.Tensor` with shape:
        `[batch_size, sequence_length, embed_size]`.
    """
    with tf.variable_scope(
        name, default_name='embedding_layer', values=[inputs]
    ):

        embedding_matrix = tf.get_variable(
            'embedding_matrix',
            initializer=tf.random_normal((vocab_size, embed_size)),
            trainable=True
        )

        return tf.nn.embedding_lookup(embedding_matrix, inputs)


def learned_positional_encoding(
    sequence_length, embed_size, name=None
):
    """
    Learned positional encoding.

    Args:
        - sequence_length: length of the sequence.
        - embed_size: size of the embedding.
        - name: optional name.
    Returns:
        A positional encoding of size `[1, sequence_length, embed_size]`.
    """
    with tf.variable_scope(
        name, default_name='learned_positional_encoding'
    ):
        return embedding_layer(
            tf.range(sequence_length), sequence_length, embed_size
        )


def sinusoidal_positional_encoding(
    sequence_length, embed_size, name=None
):
    """
    Sinusoidal positional encoding.

    Args:
        - sequence_length: length of the sequence.
        - embed_size: size of the embedding.
        - name: optional name.
    Returns:
        A positional encoding of size `[1, sequence_length, embed_size]`.
    """
    with tf.variable_scope(
        name, default_name='sinusoidal_positional_encoding'
    ):
        return common_attention.get_timing_signal_1d(
            sequence_length, embed_size
        )


def contextual_attention_layer(
    genes, smiles, attention_size, reduce_sequence=True,
    return_alphas=True, name=None
):
    """
    Inspired by Bahdanau attention, this layer implements an layer that defines
    for each token of the encoded SMILES
    (e.g. bRNN, raw embedding, conv_output) how well it targets the genes. 

    Args:
        - genes: this must be a `tf.Tensor` of shape:
            `[batch_size, num_genes]` or shape
            `[batch_size, num_genes, num_gene_features]`
            e.g. num_gene_features = 5 if copy number variation data is used.
        - smiles: encoded smiles. This must be a `tf.Tensor` of shape:
            `[batch_size, sequence_length, hidden_size]`
        - attention_size: amount of attention units (<int>).
        - reduce_sequence: whether the sequence_length dim is reduced (<bool>).
        - return_alphas: whether the attention weights are returned (<bool>).

    Returns:
        - If reduce_sequence == True (default), return will be a `tf.Tensor`
            shaped `[batch_size, hidden_size]`, else
            `[batch_size, sequence_length, hidden_size]`.
        - If return_alphas == True, return will be a tuple of 2 `tf.Tensor`,
            the first as the attention output and the second as the attention
            weights (`[batch_size, sequence_length]`).
    """
    with tf.variable_scope(
        name, default_name='merged_attention_layer',
        values=[genes, smiles]
    ):
        genes = tf.expand_dims(genes, 2) if len(genes.shape) == 2 else genes
        hidden_size = smiles.shape[2].value
        num_genes = genes.shape[1].value
        num_gene_features = genes.shape[2].value

        # Trainable parameters.
        w_num_gene_features = tf.Variable(
            tf.random_normal([num_gene_features], stddev=0.1)
        )
        w_genes = tf.Variable(
            tf.random_normal([num_genes, attention_size], stddev=0.1)
        )
        b_genes = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        
        w_smiles = tf.Variable(
            tf.random_normal([hidden_size, attention_size], stddev=0.1)
        )
        b_smiles = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('x'):
            # Applying fully connected layer with non-linear activation and
            # genes context to each of the batch_size * sequence_length.
            # Shape of `x` is `[batch_size, sequence_length, attention_size]`

            genes_collapsed = tf.tensordot(
                genes, w_num_gene_features, axes=[2, 0]
            )

            x = tf.tanh(
                    tf.expand_dims(
                        tf.tensordot(
                            genes_collapsed, w_genes, axes=1
                        ) + b_genes,
                        axis=1
                    ) 
                    + (tf.tensordot(smiles, w_smiles, axes=1) + b_smiles)
            )

        # For each of the timestamps its vector of size attention_size
        # from `v` is reduced with `u` vector
        # `[batch_size, sequence_length]`
        xv = tf.tensordot(x, v, axes=1, name='unnormalized')
        # `[batch_size, sequence_length]`
        alphas = tf.nn.softmax(xv, name='alphas')

        # If reduce_sequence is true, result is `[batch_size, hidden_size]`
        # else it is `[batch_size, sequence_length, hidden_size]`
        output = (
            tf.reduce_sum(smiles * tf.expand_dims(alphas, -1), 1)
            if reduce_sequence else
            smiles * tf.expand_dims(alphas, -1)
        )

        # Optionally return the attention weights
        return (
            (output, alphas)
            if return_alphas else
            output
        )


def contextual_attention_matrix_layer(
    genes, smiles,
    return_scores=False, name=None
):
    """
    Modifies general/multiplicative attention as defined by Luong. Computes
    a score matrix between genes and smiles, filters both with their 
    respective attention weights and returns a joint feature vector.

    Args:
        - genes: this must be a `tf.Tensor` that can be of shape:
            `[batch_size, num_genes]` or
            `[batch_size, num_genes, num_gene_features]`
            num_gene_features=1 if only transcriptomic data
            (gene expression profiles).
            are used, but num_gene_features=5 if genomic data
            (copy number variation) is also used.
        - smiles: encoded smiles. This must be a `tf.Tensor` of shape:
            `[batch_size, sequence_length, hidden_size]`.
        - return_scores: whether the unnormalized attention matrix
            is returned (<bool>).

    Returns:
        - If return_scores = False (default), return will be a
            `tf.Tensor` of shape
            `[batch_size, hidden_size + num_gene_features]`.
        - If return_scores = True, return will be two `tf.Tensor`, the second 
            carrying the unnormalized attention weights of shape 
            `[batch_size, num_genes, sequence_length]).

    NOTE: To get the molecular attention, collapse num_genes of returned
        scores, then apply softmax. Preferentially, merge across multiheads
        (and conv kernel sizes) to get final distribution.
    """
    with tf.variable_scope(
        name, default_name='attention_hypercube_layer', values=[genes, smiles]
    ):

        hidden_size = smiles.shape[2].value
        genes = tf.expand_dims(genes, 2) if len(genes.shape) == 2 else genes
        num_gene_features = genes.shape[2].value

        # cnv features treated like hidden dimension of input sequence.
        w = tf.Variable(tf.random_normal(
            [num_gene_features, hidden_size], stddev=0.1)
        )
        
        # Luong general attention. See: https://arxiv.org/pdf/1508.04025.pdf.
        # Scores has shape `[batch_size, num_genes, sequence_length]`.
        scores = tf.tanh(
            tf.matmul(
                # This has shape `[batch_size, num_genes, hidden_size]`
                tf.tensordot(genes, w, axes=(2, 0)),
                tf.transpose(smiles, (0, 2, 1))
            ), name='attention_scores'
        )
        
        # Shapes `[batch_size, sequence_length]` and `[batch_size, num_genes]`
        # respectively.
        alpha_smiles = tf.nn.softmax(
            tf.reduce_sum(scores, axis=1),
            axis=1, name='alpha_smiles'
        )
        alpha_genes = tf.nn.softmax(
            tf.reduce_sum(scores, axis=2),
            axis=1, name='alpha_genes'
        )
        filtered_smiles = tf.reduce_sum(
            smiles * tf.expand_dims(alpha_smiles, -1),
            axis=1, name='filtered_smiles'
        )
        filtered_genes= tf.reduce_sum(
            genes * tf.expand_dims(alpha_genes, -1),
            axis=1, name='filtered_genes'
        )
        outputs = tf.concat([
            filtered_smiles, filtered_genes],
            axis=1, name='outputs'
        )

        # Optionally return the attention weights.
        return (
            (outputs, scores)
            if return_scores else
            outputs
        )