import tensorflow as tf

import gpflow
from .orbits import Orbit


class Invariant(gpflow.kernels.Kernel):
    def __init__(self, basekern: gpflow.kernels.Kernel, orbit: Orbit):
        super().__init__()
        self.basekern = basekern
        self.orbit = orbit

    def K(self, X, X2=None):
        X_orbit = self.orbit(X)
        N, num_orbit_points, D = tf.shape(X_orbit)[0], tf.shape(X_orbit)[1], tf.shape(X)[1]
        X_orbit = tf.reshape(X_orbit, (-1, D))
        Xp2 = tf.reshape(self.orbit(X2), (-1, D)) if X2 is not None else None

        bigK = self.basekern.K(X_orbit, Xp2)  # [N * num_patches, N * num_patches]
        K = tf.reduce_mean(tf.reshape(bigK, (N, num_orbit_points, -1, num_orbit_points)), [1, 3])
        return K

    def K_diag(self, X):
        Xp = self.orbit(X)

        def sumbK(Xp):
            return tf.reduce_mean(self.basekern.K(Xp))

        # Can use vectorised_map?
        #return tf.vectorized_map(sumbK, Xp)
        return tf.map_fn(sumbK, Xp)
        
        # return tf.reduce_sum(tf.map_fn(self.basekern.K, Xp), [1, 2]) / self.num_patches ** 2.0


class StochasticInvariant(Invariant):
    def __init__(self, basekern: gpflow.kernels.Kernel, orbit: Orbit):
        super().__init__(basekern, orbit)

    def K(self, X, X2=None):
        Xp = self.orbit(X)
        orbit_dim = Xp.shape[2]
        Xp = tf.reshape(Xp, (-1, orbit_dim))
        Xp2 = tf.reshape(self.orbit(X2), (-1, orbit_dim)) if X2 is not None else None

        bigK = self.basekern.K(Xp, Xp2)
        bigK = tf.reshape(bigK, (tf.shape(X)[0], self.orbit.minibatch_size, -1, self.orbit.minibatch_size))

        if self.orbit.minibatch_size < self.orbit.orbit_size:
            bigKt = tf.transpose(bigK, (0, 2, 1, 3))  # N x N2 x M x M
            diag_sum = tf.reduce_sum(tf.linalg.diag_part(bigKt), 2)
            edge_sum = tf.reduce_sum(bigKt, (2, 3)) - diag_sum

            if self.orbit.orbit_size < float("inf"):
                return (edge_sum * self.w_edge + diag_sum * self.w_diag)  # / self.orbit.orbit_size ** 2.0
            elif self.orbit.orbit_size == float("inf"):
                return edge_sum / (self.orbit.minibatch_size * (self.orbit.minibatch_size - 1))
            else:
                raise ValueError
        elif self.orbit.minibatch_size == self.orbit.orbit_size:
            return tf.reduce_mean(bigK, [1, 3])
        else:
            raise ValueError

    def K_diag(self, X):
        Xp = self.orbit(X)

        def sumbK(Xp):
            K = self.basekern.K(Xp)  # C x C
            if self.orbit.minibatch_size < self.orbit.orbit_size:
                diag_sum = tf.reduce_sum(tf.linalg.diag_part(K))
                edge_sum = tf.reduce_sum(K) - diag_sum

                return edge_sum * self.w_edge + diag_sum * self.w_diag
            elif self.orbit.minibatch_size == self.orbit.orbit_size:
                return tf.reduce_mean(K)

        return tf.map_fn(sumbK, Xp)
        # return tf.reduce_sum(tf.map_fn(self.basekern.K, Xp), [1, 2]) / self.num_patches ** 2.0
        

    @property
    def w_diag(self):
        return 1.0 / (self.orbit.minibatch_size * self.orbit.orbit_size)

    @property
    def w_edge(self):
        return (1.0 - 1.0 / self.orbit.orbit_size) / (self.orbit.minibatch_size * (self.orbit.minibatch_size - 1))

    @property
    def mw_diag(self):
        return 1.0 / self.orbit.orbit_size - self.mw_full / self.orbit.minibatch_size

    @property
    def mw_full(self):
        return self.orbit.minibatch_size / (self.orbit.minibatch_size - 1) * (1.0 - 1.0 / self.orbit.orbit_size)
