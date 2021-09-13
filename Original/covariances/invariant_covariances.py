import tensorflow as tf

from gpflow.base import TensorLike
from gpflow.covariances.dispatch import Kuf, Kuu
from ..inducing_variables import ConvolvedInducingPoints, StochasticConvolvedInducingPoints
from ..kernels import Invariant, StochasticInvariant


#@Kuu.register(ConvolvedInducingPoints, Invariant)
@Kuu.register(StochasticConvolvedInducingPoints, StochasticInvariant)
def Kuu_convip_invariant(inducing_variable: StochasticConvolvedInducingPoints, kernel: StochasticInvariant, *, jitter=0.0):
    Kzz = kernel.basekern.K(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

@Kuu.register(ConvolvedInducingPoints, Invariant)
def Kuu_convip_invariant2(inducing_variable: ConvolvedInducingPoints, kernel: Invariant, *, jitter=0.0):
    Kzz = kernel.basekern.K(inducing_variable.Z)
    #print(Kzz.shape)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz


@Kuf.register(ConvolvedInducingPoints, Invariant, TensorLike)
def Kuf_convip_invariant(inducing_variable, kern, Xnew):
    N, M = tf.shape(Xnew)[0], tf.shape(inducing_variable.Z)[0]
    orbit_size = kern.orbit.orbit_size
    Xorbit = kern.orbit(Xnew)  # [N, orbit_size, D]
    Kzx_orbit = kern.basekern.K(inducing_variable.Z, tf.reshape(Xorbit, (N * orbit_size, -1)))  # [M, N * orbit_sz]
    Kzx = tf.reduce_mean(tf.reshape(Kzx_orbit, (M, N, orbit_size)), [2])
    #print(Kzx.shape)
    return Kzx


@Kuf.register(StochasticConvolvedInducingPoints, StochasticInvariant, object)
def Kuf_stochastic_convip_stochastic_invariant(inducing_variable, kern, Xnew):
    """
    :return: [M, N, minibatch_size]
    """
    N, M = tf.shape(Xnew)[0], tf.shape(inducing_variable.Z)[0]
    Xorbit = tf.reshape(kern.orbit(Xnew), (N * kern.orbit.minibatch_size, -1))  # [N * minibatch_size, D]
    # tf.print('shapes: M, N, Xorbit', M, N, Xorbit.shape)
    Kzx = kern.basekern.K(inducing_variable.Z, Xorbit)  # [M, N * minibatch_size]
    return tf.reshape(Kzx, (M, N, kern.orbit.minibatch_size))
