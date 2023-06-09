import math
import numpy as np
import tensorflow as tf


class Kernel(object):
    """A class representing a kernel function."""

    def __call__(self, X):
        """Returns the Gram matrix for the given data.
        Args:
            X: An (n, d) matrix, consisting of n data points with d dimensions.
        Returns:
            An (n, n) matrix where the (i, j)th entry is k(x_i, x_j).
        """

        pass

    def random_features(self, X, D):
        """Returns a random Fourier feature map for the given data, following
        the approach in Rahimi and Recht [1].

        [1] https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

        Args:
            X: An (n, d) matrix, consisting of n data points with d dimensions.
            D (int): The number of features in the feature map.
        Returns:
            An (n, D) matrix for which the inner product of row i and row j is
            approximately k(x_i, x_j).
        """

        raise NotImplementedError


class LinearKernel(Kernel):
    """The linear kernel, defined by the inner product
        k(x_i, x_j) = x_i^T x_j.
    """

    def __call__(self, X):
        return tf.matmul(X, X, transpose_b=True)


class PolynomialKernel(Kernel):
    """The polynomial kernel, defined by
        k(x_i, x_j) = (a x_i^T x_j + b)^d
    with coefficient a, bias b, and degree d.
    """

    def __init__(self, a, b, d):
        self.a = a
        self.b = b
        self.d = d

    def __call__(self, X):
        return (self.a * tf.matmul(X, X, transpose_b=True) + self.b) ** self.d


class LaplacianKernel(Kernel):
    """The Laplacian kernel, defined by
        k(x_i, x_j) = exp(-||x_i - x_j||_1 / sigma)
    with bandwidth parameter sigma.
    """

    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        X_rowdiff = tf.expand_dims(X, 1) - tf.expand_dims(X, 0)
        return tf.exp(-tf.reduce_sum(input_tensor=tf.abs(X_rowdiff), axis=2) / self.sigma)


class GaussianKernel(Kernel):
    """The Gaussian kernel, defined by
        k(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 sigma^2))
    with bandwidth parameter sigma.
    """

    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        X_rowdiff = tf.expand_dims(X, 1) - tf.expand_dims(X, 0)
        return tf.exp(-tf.reduce_sum(input_tensor=X_rowdiff ** 2, axis=2) / (2 * self.sigma ** 2))

    # Overwrite this function
    def random_features(self, X, D):
        # RFF
        omega = tf.random.normal(
            tf.stack([tf.shape(input=X)[1], D]), stddev=1.0 / self.sigma, dtype=X.dtype)
        b = tf.random.uniform([D], maxval=2 * math.pi, dtype=X.dtype)
        return math.sqrt(2.0 / D) * tf.cos(tf.matmul(X, omega) + b)


class vonMisesKernel(Kernel):
    """
    The von-Mises kernel on the sphere, defined by
        k(x_i, x_j) = exp(-<x_i, x_j>^2 / 2 sigma^2)
    Here sigma will be chosen as the median of the <x_i, x_j> values.
    """
    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        inner_prods = tf.matmul(X, X, transpose_b=True)
        angles = tf.math.acos(inner_prods) ** 2
        return tf.exp(-angles / (2 * self.sigma ** 2))

    def random_features(self, X, D):
        raise NotImplementedError("Not yet developed")


class EqualityKernel(Kernel):
    """The equality kernel, defined by
        k(x_i, x_j) = f(x_i_1 == x_j_1, x_i_2 == x_j_2, ...)
    where f is either "mean" or "product".
    """

    def __init__(self, composition="product"):
        assert composition in ("mean", "product")
        self.composition = composition

    def __call__(self, X):
        X_equal = tf.cast(tf.equal(tf.expand_dims(X, 0), tf.expand_dims(X, 1)), dtype=tf.float64)
        reduce = {
            "mean": tf.reduce_mean,
            "product": tf.reduce_prod
        }[self.composition]
        return reduce(X_equal, reduction_indices=2)


def get_kernel(kernelname, *args, **kwargs):
    if kernelname == "linear":
        return LinearKernel()
    elif kernelname == "gaussian":
        return GaussianKernel(*args)
    elif kernelname == "laplacian":
        return LaplacianKernel(*args)
    elif kernelname == "polynomial":
        # Keep the input order of a, b, d; (a x_i^T x_j + b)^d
        return PolynomialKernel(*args)  # unpack the tuple
    elif kernelname == "vonMises":
        return vonMisesKernel(*args)
    else:
        raise ValueError("Name of ker should be specified accurately")


if __name__ == "__main__":
    a = get_kernel("gaussian", 1)
    print(np.log(a([[1, 2], [3, 3]])))

