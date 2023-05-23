"""
Source code for kernel variable selection of compositional data via amalgamation.
Example usage is provided at the bottom of this file.

Note: We borrowed some codes from https://github.com/Jianbo-Lab/CCM
      The name ccm comes from "conditional covariance minimization."
"""

import numpy as np
import pandas as pd
import kernel
from scipy.spatial.distance import pdist
import tensorflow as tf


def center(X):
    """Returns the centered version of the given square matrix, namely

        (I - (1/n) 1 1^T) X (I - (1/n) 1 1^T)
            = X - (1/n) 1 1^T X - (1/n) X 1 1^T + (1/n^2) 1 1^T X 1 1^T.

    Args:
        X: An (n, n) matrix.

    Returns:    
        The row- and column-centered version of X.
    """

    mean_col = tf.reduce_mean(X, axis=0, keepdims=True)
    mean_row = tf.reduce_mean(X, axis=1, keepdims=True)
    mean_all = tf.reduce_mean(X)
    return X - mean_col - mean_row + mean_all


def project(v, z):
    """Returns the Euclidean projection of the given vector onto the positive
    simplex, i.e. the set

        {w : \sum_i w_i = z, w_i >= 0}.

    Implements the core given in Figure 1 of Duchi et al. (2008) [1].

    [1] http://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    Args:
        v: A vector, clipped in advance (via clip_and_project function)
        z: The desired sum of the components. Must be strictly positive.

    Returns:
        The Euclidean projection of v onto the positive simplex of size z.
    """
    
    assert z > 0

    z = tf.convert_to_tensor(z, dtype=tf.float32)

    mu = tf.sort(v, direction='DESCENDING')
    mu_cumsum = tf.cumsum(mu)
    max_index = tf.where(tf.greater(mu * np.arange(1, len(v) + 1), mu_cumsum - z))[-1][0]
    temp = tf.cast(max_index + 1, dtype=tf.float32)
    theta = (mu_cumsum[max_index] - z) / temp
    return tf.maximum(v - theta, 0)


@tf.function
def clip_and_project(w, num_features):
    """
     clip and project w onto the l1-ball, sum(w) <= num_features
     makes sure the w have values inside the interval [0, 1]
    """
    w = tf.clip_by_value(w, 0, 1)

    # If sum(w) > num_features, project to the l1-ball
    if tf.greater(tf.reduce_sum(w), num_features):
        w = project(w, num_features)
    return w


class CCM(object):
    """
    Initialization and evaluation of the objective loss function (after continuous relaxation)
    """

    def __init__(
            self, X, Y, num_features, transform_Y,
            ker="gaussian", init=None):
        """
        :param X: Input data (np.ndarray or pd.DataFrame)
        :param Y: Label of data (np.ndarray)
        :param transform_Y: None, "one-hot", "binary" (one of these three)
        :param ker: Choice of kernels (gaussian, laplacian, vonMises, ...)
        :param init: Customized initialization of gradient descent (use this if you want to apply other initializations)
        """

        assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame)
        assert (np.abs(X.sum(axis=1) - 1) < 1e-8).all(), 'X is not compositional'
        assert isinstance(Y, np.ndarray)
        assert X.ndim == 2
        assert Y.ndim == 1

        X = np.array(X)
        n, d = X.shape
        assert Y.shape == (n,)
        self.d = d
        self.n = n

        # X as a class element
        self.X = X

        # Deriving kernel parameters
        self.sigma = tf.Variable(np.median(pdist(X)) / np.sqrt(2), dtype=tf.float32, trainable=False)
        self.kernel_X = kernel.get_kernel(ker, self.sigma)

        # Preprocess for Y
        assert transform_Y in (None, "binary", "one-hot")

        if transform_Y == "binary":
            values = sorted(set(Y.ravel()))
            assert len(values) == 2
            Y_new = np.zeros(n)
            # Y is going to be centered
            Y_new[Y == values[0]] = -1
            Y_new[Y == values[1]] = 1
            Y = Y_new

        elif transform_Y == "one-hot":
            values = sorted(set(Y.ravel()))
            Y_new = np.zeros((n, len(values)))
            for i, value in enumerate(values):
                Y_new[Y == value, i] = 1
            Y = Y_new

        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        # Column-wise centering.
        self.Y = tf.convert_to_tensor(Y - Y.mean(0), dtype=tf.float32)

        # Weight variable w.r.t. projected gradient descent
        self.w = tf.Variable(project(tf.ones(d), num_features), dtype=tf.float32,
                 constraint=lambda x: clip_and_project(x, num_features))
        if isinstance(init, np.ndarray):
            assert init.size == d
            self.w.assign(init)
        elif init == "zero":
            self.w.assign(np.zeros(d))

    # Empirical estimate of Tr[Cond_cov], i.e., the loss of the problem
    def __call__(self, epsilon, D_approx=None):
        # Amalgamation of compositional data X_w
        amalgam = tf.concat([self.X * self.w,
                             (1 - tf.math.reduce_sum(self.X * self.w, axis=1))[:, tf.newaxis]], axis=1)

        # No RFF approximation
        if D_approx is None:
            G_X_w = center(self.kernel_X(amalgam))

            G_X_w += self.n * epsilon * tf.eye(self.n)
            G_X_w_inv = tf.linalg.inv(G_X_w)

        # RFF approximation
        # Some dtype mismatches live here.. to be modified.
        else:
            U_w = self.kernel_X.random_features(amalgam, D_approx)

            V_w = tf.linalg.matmul(
                tf.subtract(
                    tf.eye(self.n, dtype=tf.float32),
                    tf.divide(
                        tf.ones((self.n, self.n), dtype=tf.float32),
                        tf.constant(float(self.n), dtype=tf.float32))),
                U_w)

            # eq. 21, arXiv:1707.01164, omitting constant term

            G_X_w_inv = tf.linalg.matmul(
                tf.scalar_mul(
                    tf.constant(-1.0, dtype=tf.float32),
                    V_w),
                tf.linalg.matmul(
                    tf.linalg.inv(
                        tf.add(
                            tf.linalg.matmul(
                                V_w,
                                V_w,
                                transpose_a=True),
                            tf.multiply(
                                tf.constant(self.n * epsilon, dtype=tf.float32),
                                tf.eye(D_approx, dtype=tf.float32)))),
                    V_w,
                    transpose_b=True))

        return tf.linalg.trace(tf.linalg.matmul(self.Y, tf.linalg.matmul(G_X_w_inv, self.Y), transpose_a=True))


def train_step(model, optimizer, epsilon=0.1, D_approx=None):
    with tf.GradientTape() as tape:
        loss = model(epsilon=epsilon, D_approx=D_approx)
    gradients = tape.gradient(loss, [model.w])
    optimizer.apply_gradients(zip(gradients, [model.w]))

    return loss, model.kernel_X.sigma
    

def ccm(X, Y, num_features, type_Y, epsilon, learning_rate=0.001,
        iterations=1000, D_approx=None, verbose=True, init=None, kernel="gaussian", 
        optimizer=None, continuation=False):
    """
    This function carries out variable selection of compositional data.
    Args:
        X: An (n, d) numpy array or pandas DataFrame
        Y: An (n,) numpy array.

        num_features: int. Number of selected features.

        type_Y: str. Type of the response variable.
                Possible choices: 'ordinal','binary','categorical','real-valued'.

        learning_rate: learning rate of projected gradient method. Used if optimizer=None.

        iterations: number of iterations for optimization.

        D_approx: optional, rank-D kernel approximation via RFF

        verbose: print the loss every 100 updates.

        init: initialization of the gradient descent. By default, it is the center of the simplex.
                Possible choices: None, "zero", or customized np.ndarray.

        kernel: choice of kernels (gaussian, laplacian, vonMises, ...)

        optimizer: Plain GD as default. 
                Possible choices: Adam, or AMSGrad.. tf.keras.optimizers objects for acceleration.
        
        continuation: continuation method to avoid local minima (not recommended)

    Return:
        ranks: An (d,) numpy array, indicates feature ranks
        w: resulting weights
    """
    assert type_Y in ('ordinal', 'binary', 'categorical', 'real-valued')
    if type_Y == 'ordinal' or type_Y == 'real-valued':
        transform_Y = None
    elif type_Y == 'binary':
        transform_Y = 'binary'
    elif type_Y == 'categorical':
        transform_Y = 'one-hot'

    fs = CCM(X, Y, num_features, transform_Y, ker=kernel, init=init)

    if continuation:
        # fs.sigma.assign(1.2 * fs.sigma)
        sigma_evolution = fs.sigma * 1. / (tf.sqrt(2.0) * iterations)
    
    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # For several training objects
    train = tf.function(train_step)

    # Training
    for i in range(1, iterations + 1):
        loss, sigma = train(fs, epsilon=epsilon, optimizer=optimizer, D_approx=D_approx)

        if verbose and i % 100 == 0:
            print("\niteration", i, "loss", loss.numpy(), end='')
            if continuation:
                print(" // sigma:", sigma.numpy(), end='')
        # Continuation
        if continuation:
            fs.sigma.assign_sub(sigma_evolution)
        # If you want to perform early stopping, write down that code here
        
    print("Train finished")

    # Results
    w = fs.w.numpy()
    # if verbose:
    #     print('The weights on features are: ', w)

    idx = np.random.permutation(fs.d)
    permutated_weights = w[idx]
    permutated_ranks = (-permutated_weights).argsort().argsort() + 1
    ranks = permutated_ranks[np.argsort(idx)]

    return ranks, w


if __name__ == "__main__":
    """
    Example usage: selection of differentially abundant variables
    """
    a1, a2 = np.ones(10), np.ones(10)
    
    # Signal at indices 2, 3 (not very strong)
    a2[2:4] += 2

    # Generate compositional data with differential abundance
    np.random.seed(777)
    X1, Y1 = np.random.dirichlet(a1, 50), np.zeros(50)
    X2, Y2 = np.random.dirichlet(a2, 50), np.ones(50)
    X = np.r_[X1, X2]
    Y = np.r_[Y1, Y2]

    # Perform variable selection
    # learning rate is chosen to show reduction in loss clearly.
    rank, w = ccm(X, Y, num_features=2, type_Y="binary", epsilon=0.001,
                learning_rate=3e-5, iterations=2000)
    print("Selected variables:", np.sort(np.argsort(rank)[:2]))
    print("Weights:", w)
