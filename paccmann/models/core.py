"""Core for paccmann models."""
import tensorflow as tf
from ..hyperparams import LOSS_FN_FACTORY, OPTIMIZER_FACTORY
from ..metrics import pearson


def paccmann_model_fn(
    features, labels, mode, params, model_specification_fn
):
    """
    Wrapper for IC50 prediction.

    Usage example:
        Pass it as model_fn parameter into an estimator:
        ```
        estimator = tf.estimator.Estimator(
            model_fn=(
                lambda features, labels, mode, params: paccmann_model_fn(
                    features, labels, mode, params,
                    model_specification_fn=custom_model_specification_fn
                )
            ),
            model_dir=model_dir,
            params=params
        )
        ```
    Args:
        - features: features for the observations (<dict<string, tf.Tensor>>).
        - labels: labels associated (<tf.Tensor>).
        - mode: mode for the model (<tf.estimator.ModeKeys>).
        - params: parameters for the model (<dict<string, object>>).
        - model_specification_fn: model specification function (<function>).
            A function with the following signature:
            (
                features, labels, mode, params
            ) -> (
                predictions (<tf.Tensor>),
                prediction_dict (<dict<string, tf.Tensor>>)
            )
    Returns:
        Estimator specifications (<tf.estimator.EstimatorSpec>).
    """
    # NOTE: run model specification function.
    predictions, prediction_dict = model_specification_fn(
        features, labels, mode, params
    )
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=prediction_dict
        )
    
    # True molar IC50 for convenience during analysis.
    prediction_dict.update({
        'True_IC50_micromolar': tf.exp(
            labels*(
                params.get('max', 0.0) - params.get('min', 0.0)
            ) + params.get('min', 0.0)
        )
    })
    # Set the loss function (defaults to MSE).
    loss_fn = LOSS_FN_FACTORY[params.get('loss_function', 'mse')]
    loss = loss_fn(labels, predictions)
    pearson_correlation = pearson(labels, predictions)

    # Metrics.
    metrics_mse = tf.metrics.mean_squared_error(
        labels=labels, predictions=predictions
    )
    metrics_pearson = tf.contrib.metrics.streaming_pearson_correlation(
        labels=labels, predictions=predictions
    )

    # EstimatorSpec requires values to be tuples (loss, update_op),
    # so wrap our loss into `tf.metrics.mean`.
    metric_ops = {
        'mse': metrics_mse,
        'pearson': metrics_pearson,
    }

    optimizer_class = OPTIMIZER_FACTORY[params.get('optimizer', 'adam')]
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('train_pearson', pearson_correlation)
        tf.summary.scalar('train_loss', loss)
        learning_rate = tf.train.exponential_decay(
            params.get('learning_rate', 0.001),
            tf.train.get_global_step(),
            decay_steps=params.get('decay_steps', 3000),
            decay_rate=params.get('decay_rate', 0.96)
        )
        optimizer = optimizer_class(
            learning_rate=learning_rate
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op,
            eval_metric_ops=metric_ops
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss,
            eval_metric_ops=metric_ops
        )
