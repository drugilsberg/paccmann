"""Customizable model hyperparameter """
import tensorflow as tf
from .layers import (
    dense_attention_layer, sequence_attention_layer,
    contextual_attention_layer, learned_positional_encoding,
    sinusoidal_positional_encoding
)


# Specifies types of RNN cells used for the SMILES encoder.
# Uses CUDA implementation (GPU) if available.
RNN_CELL_FACTORY = {
    'lstm': (
        tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        if tf.test.is_gpu_available() else
        tf.contrib.rnn.LSTMBlockCell
    ),
    'gru': (
        tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell
        if tf.test.is_gpu_available() else
        tf.contrib.rnn.GRUCell
    )
}

LOSS_FN_FACTORY = {
    'mse': tf.losses.mean_squared_error
}

ACTIVATION_FN_FACTORY = {
    'relu': tf.nn.relu,
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'elu': tf.nn.elu,
    'selu': tf.nn.selu,
    'leaky_relu': tf.nn.leaky_relu,
    'softmax': tf.nn.softmax
}

OPTIMIZER_FACTORY = {
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagradda': tf.train.AdagradDAOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'ftlr': tf.train.FtrlOptimizer,
    'gd': tf.train.GradientDescentOptimizer,
    'pgd': tf.train.ProximalGradientDescentOptimizer,
    'padagrad': tf.train.ProximalAdagradOptimizer
}

ATTENTION_FN_FACTORY = {
    'dense': dense_attention_layer,
    'sequence': sequence_attention_layer,
    'contextual': contextual_attention_layer
}

POSITIONAL_ENCODING_FACTORY = {
    'learned': learned_positional_encoding,
    'sinusoidal': sinusoidal_positional_encoding
}
