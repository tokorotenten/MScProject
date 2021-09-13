import tensorflow as tf
from tensorflow import linalg as tfla

from gpflow.conditionals.dispatch import conditional
from gpflow.config import default_jitter
from .. import covariances
from ..inducing_variables import StochasticConvolvedInducingPoints
from ..kernels import StochasticInvariant


def sub_conditional(Kmn, Kmm, fKnn, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Adapted version of base_conditional. Need this because things like Lm do not get memoised yet.
    :param Kmn: M x N x C
    :param Kmm: M x M
    :param fKnn: N x C x C
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    print("sub conditional")
    # Things to start
    N, M, C = tf.shape(Kmn)[1], tf.shape(Kmn)[0], tf.shape(Kmn)[2]
    if full_cov:
        raise NotImplementedError

    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R
    Lm = tfla.cholesky(Kmm)

    # Compute the projection matrix A
    dA = tfla.triangular_solve(Lm, tf.reshape(Kmn, (M, N * C)), lower=True)  # M x NC
    sA = tf.reduce_mean(tf.reshape(dA, (M, N, C)), 2)  # M x N

    # compute the covariance due to the conditioning
    mKnn = tf.reduce_mean(fKnn, (1, 2))
    dKnn = tf.reshape(tfla.diag_part(fKnn), (N * C,))
    sfvar, dfvar = [Knn - tf.reduce_sum(tf.square(A), 0) for A, Knn in zip([sA, dA], [mKnn, dKnn])]  # R x N
    sfvar, dfvar = [tf.tile(fvar[None, :], [num_func, 1]) for fvar in [sfvar, dfvar]]  # R x N

    # another backsubstitution in the unwhitened case
    if not white:
        sA, dA = [tfla.triangular_solve(tf.transpose(Lm), A, lower=False) for A in [sA, dA]]

    # construct the conditional mean
    sfmean, dfmean = [tf.matmul(A, f, transpose_a=True) for A in [sA, dA]]

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            sLTA, dLTA = [A * tf.expand_dims(tf.transpose(q_sqrt), 2) for A in [sA, dA]]  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tfla.band_part(q_sqrt, -1, 0)  # R x M x M
            sA_tiled, dA_tiled = [tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1])) for A in [sA, dA]]
            sLTA, dLTA = [tf.matmul(L, A_tiled, transpose_a=True) for A_tiled in [sA_tiled, dA_tiled]]  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        sfvar, dfvar = [fvar + tf.reduce_sum(tf.square(LTA), 1)
                        for fvar, LTA in zip([sfvar, dfvar], [sLTA, dLTA])]  # R x N

    if not full_cov:
        sfvar, dfvar = [tf.transpose(fvar) for fvar in [sfvar, dfvar]]  # N x R

    return sfmean, sfvar, dfmean, dfvar  # N x R, N x R, NC x R, NC x R


@conditional.register(object, StochasticConvolvedInducingPoints, StochasticInvariant, object)
def stochastic_inv_conditional(Xnew, inducing_variable, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                               white=False):
    print("stochastic_inv_conditional")
    if full_output_cov:
        # full_output_cov is misused here
        raise ValueError("Can not handle `full_output_cov`.")
    if full_cov:
        raise ValueError("Can not handle `full_cov`.")
    Kuu = covariances.Kuu(inducing_variable, kern, jitter=default_jitter())
    Kuf = covariances.Kuf(inducing_variable, kern, Xnew)  # M x N x C  where C is the "orbit minibatch" size
    Xp = kern.orbit(Xnew)  # N x C x D
    Knn = tf.map_fn(lambda X: kern.basekern.K(X), Xp)  # N x C x C

    est_fmu, full_fvar_mean, fmu, fvar = sub_conditional(Kuf, Kuu, Knn, f, q_sqrt=q_sqrt, white=white)
    # est_fmu, full_fvar_mean = base_conditional(tf.reduce_mean(Kuf, 2), Kuu, tf.reduce_mean(Knn, (1, 2)), f,
    #                                            full_cov=False, q_sqrt=q_sqrt, white=white)
    # fmu, fvar = base_conditional(tf.reshape(Kuf, (M, N * C)), Kuu, tf.reshape(tf.matrix_diag_part(Knn), (N * C,)), f,
    #                              full_cov=False, q_sqrt=q_sqrt, white=white)  # NC x R
    # N x R, 𝔼[est[μ]] = μ -- predictive mean

    M, N, C = tf.shape(Kuf)[0], tf.shape(Kuf)[1], tf.shape(Kuf)[2]
    diag_fvar_mean = tf.reduce_mean(tf.reshape(fvar, (N, C, -1)), 1)  # N x R
    est_fvar = full_fvar_mean * kern.mw_full + diag_fvar_mean * kern.mw_diag
    # N x R, 𝔼[est[σ²]] = σ² -- predictive variance

    diag_fmu2_mean = tf.reduce_mean(tf.reshape(fmu ** 2.0, (N, C, -1)), 1)
    est_fmu2_minus = est_fmu ** 2.0 * (kern.mw_full - 1.0) + diag_fmu2_mean * kern.mw_diag

    # N x R est[μ²] - est[μ]²
    est2 = est_fvar + est_fmu2_minus

    # We return:
    # - est[μ],                      such that 𝔼[est[μ]] = μ
    # - est[σ²] + est[μ²] - est[μ]², such that 𝔼[est[σ²] + est[μ²] - est[μ]²] = σ² + μ² - 𝔼[est[μ]²]
    # This ensures that the Gaussian likelihood gives an unbiased estimate for the variational expectations.
    return est_fmu, est2
