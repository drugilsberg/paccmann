"""Tensorflow learning utilities."""
import tensorflow as tf
from .datasets import (
    generate_dataset, feature_name_to_placeholder, validate_feature_names,
    record_parser
)


def train_input_fn(params):
    """
    Train function given params for paccmann.

    Args:
        - params: train parameters (<dict>).
            A dictionary with the following mandatory fields:
            - batch_size: size of the batch (<int>).
            - feature_names: list of feature to consider (<list<string>>).
            - train_filepath: path to training file (<string>).
            Optional:
            - buffer_size: size of the buffer for data shuffling (<int>).
    Returns:
        A tuple containing features and labels.
    """
    batch_size = params['batch_size']
    filepath = params['train_filepath']
    feature_names = validate_feature_names(params['feature_names'])
    dataset = generate_dataset(
        filepath,
        buffer_size=params.get('reader_buffer_size', int(256e+6)),
        num_parallel_reads=params.get('number_of_threads', None)
    ).apply(
        tf.contrib.data.shuffle_and_repeat(
            buffer_size=params.get('buffer_size', 20000),
            count=params.get('epochs', None)
        )
    ).apply(
        tf.contrib.data.map_and_batch(
            map_func=lambda record: record_parser(
                record, params, feature_names=feature_names
            ),
            batch_size=batch_size,
            num_parallel_batches=params.get('number_of_threads', 1),
            drop_remainder=params.get('drop_remainder', True)
        )
    ).prefetch(
        buffer_size=params.get(
            'prefetch_buffer_size',
            max(batch_size // 10, 1)
        )
    )
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


def eval_input_fn(params):
    """
    Eval function given params for paccmann.
    
    Args:
        - params: eval parameters (<dict>).
            A dictionary with the following mandatory fields:
            - batch_size: size of the batch (<int>).
            - feature_names: list of feature to consider (<list<string>>).
            - eval_filepath: path to training file (<string>).
    Returns:
        A tuple containing features and labels.
    """
    batch_size = params.get('eval_batch_size', params['batch_size'])
    filepath = params['eval_filepath']
    feature_names = validate_feature_names(params['feature_names'])
    dataset = generate_dataset(
        filepath,
        buffer_size=params.get('reader_buffer_size', int(256e+6)),
        num_parallel_reads=params.get('number_of_threads', None)
    ).apply(
        tf.contrib.data.map_and_batch(
            map_func=lambda record: record_parser(
                record, params, feature_names=feature_names
            ),
            batch_size=batch_size,
            num_parallel_batches=params.get('number_of_threads', 1),
            drop_remainder=params.get('drop_remainder', False)
        )
    ).prefetch(
        buffer_size=params.get(
            'prefetch_buffer_size',
            max(batch_size // 10, 1)
        )
    )
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


def serving_input_with_params_fn(params):
    """
    Serving function given params for paccmann.
    Args:
        - params: eval parameters (<dict>).
            A dictionary with the following mandatory fields:
            - feature_names: list of feature to consider (<list<string>>).
    Returns:
        A tf.estimator.export.ServingInputReceiver.

    """
    feature_names = validate_feature_names(params['feature_names'])
    features = {
        feature_name: feature_name_to_placeholder(feature_name, params)
        for feature_name in feature_names
    }
    return tf.estimator.export.ServingInputReceiver(features, features)