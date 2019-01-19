"""Metrics used in paccmann."""
import tensorflow as tf


def pearson(x, y):
    """
    Compute Pearson correlation.

    Args:
        - x: a `tf.Tensor`.
        - y: a `tf.Tensor`.
    Returns:
        Pearson correlation coefficient.
    """
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.reduce_sum(tf.multiply(xm, ym))
    r_den = tf.sqrt(
        tf.multiply(
            tf.reduce_sum(tf.square(xm)), tf.reduce_sum(tf.square(ym))
        )
    )
    r = r_num / r_den
    return tf.maximum(tf.minimum(r, 1.0), -1.0)