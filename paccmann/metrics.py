"""Metrics used in paccmann."""
import tensorflow as tf
import numpy as np
from scipy import linalg

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


def pearson_sklearn(x, y):
    """
    pearsonr implementation from sklearn:
    https://github.com/scipy/scipy/blob/v1.3.1/scipy/stats/stats.py#L3266-L3448

    """

    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')

    if n < 2:
        raise ValueError('x and y must have length at least 2.')

    x = np.asarray(x)
    y = np.asarray(y)

    # If an input is constant, the correlation coefficient is not defined.
    if (x == x[0]).all() or (y == y[0]).all():
        raise ValueError("Constant input, r is not defined.")

    # dtype is the data type for the calculations.  This expression ensures
    # that the data type is at least 64 bit floating point.  It might have
    # more precision if the input is, for example, np.longdouble.
    dtype = type(1.0 + x[0] + y[0])

    if n == 2:
        return dtype(np.sign(x[1] - x[0])*np.sign(y[1] - y[0])), 1.0

    xmean = x.mean(dtype=dtype)
    ymean = y.mean(dtype=dtype)

    # By using `astype(dtype)`, we ensure that the intermediate calculations
    # use at least 64 bit floating point.
    xm = x.astype(dtype) - xmean
    ym = y.astype(dtype) - ymean

    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)

    threshold = 1e-13
    if normxm < threshold*abs(xmean) or normym < threshold*abs(ymean):
        # If all the values in x (likewise y) are very close to the mean,
        # the loss of precision that occurs in the subtraction xm = x - xmean
        # might result in large errors in r.
        raise ValueError("Unstable estimate (little variance)")

    r = np.dot(xm/normxm, ym/normym)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)

    return r