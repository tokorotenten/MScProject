import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import gpflow
from gpflow.config import default_float
from gpflow.utilities.bijectors import positive
from .image_transforms import rotate_img_angles, rotate_img_angles_stn, apply_stn_batch, _stn_theta_vec, \
    apply_stn_batch_colour
from .transformer import spatial_transformer_network as stn


class Orbit(gpflow.base.Module):
    def __init__(self, orbit_size, minibatch_size=None, name=None):
        super().__init__(name=name)
        self._orbit_size = orbit_size
        self.minibatch_size = minibatch_size if minibatch_size is not None else orbit_size

    @property
    def orbit_size(self):
        return self._orbit_size

    def orbit_full(self, X):
        raise NotImplementedError

    def orbit_minibatch(self, X):
        full_orbit = tf.transpose(self.orbit_full(X), [1, 0, 2])  # [orbit_size, X.shape[0], ...]
        return tf.transpose(tf.random.shuffle(full_orbit)[:self.minibatch_size, :, :], [1, 0, 2])

    def __call__(self, X):
        if self.minibatch_size == self.orbit_size:
            return self.orbit_full(X)
        else:
            return self.orbit_minibatch(X)


class SwitchXY(Orbit):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def orbit_full(self, X):
        X_switch = tf.gather(X, [1, 0], axis=1)
        return tf.concat([X[:, None, :], X_switch[:, None, :]], axis=1)
    
class Reflect(Orbit):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def orbit_full(self, X):
        X_switch = -X
        return tf.concat([X[:, None, :], X_switch[:, None, :]], axis=1)


class GaussianNoiseOrbit(Orbit):
    def __init__(self, variance=1.0, minibatch_size=10, **kwargs):
        super().__init__(np.inf, minibatch_size, **kwargs)
        self.variance = variance

    def orbit_minibatch(self, X):
        return X[:, None, :] + tf.random.normal((X.shape[0], self.minibatch_size, X.shape[1]),
                                                stddev=self.variance ** 0.5, dtype=X.dtype)


class ImageOrbit(Orbit):
    def __init__(self, orbit_size, input_dim=None, img_size=None, minibatch_size=None, **kwargs):
        super().__init__(orbit_size, minibatch_size=minibatch_size, **kwargs)
        if input_dim is not None and img_size is None:
            print('img dim is', input_dim)
            img_size = int(tf.cast(input_dim, default_float()) ** 0.5)
        elif input_dim is None and img_size is not None:
            input_dim = img_size ** 2
        elif input_dim is not None and img_size is not None:
            assert self._img_size ** 2 == self._input_dim
        self._img_size = img_size
        self._input_dim = input_dim

    def input_dim(self, X):
        # X can be None if not required
        if self._input_dim is not None:
            return self._input_dim
        else:
            return tf.shape(X)[1]

    def img_size(self, X):
        # X can be None if not required
        if self._img_size is not None:
            return self._img_size
        else:
            return tf.cast(tf.cast(self.input_dim(X), tf.float32) ** 0.5, tf.int32)


class ImageRot90(ImageOrbit):
    """
    ImageRot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(4, input_dim=input_dim, img_size=img_size, **kwargs)

    def orbit_full(self, X):
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim(X)))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, self.input_dim(X)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim(X)))
        return tf.concat((X[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)


class ImageRotQuant(ImageOrbit):
    """
    ImageRotQuant
    Kernel invariant to any quantised rotations of the input image.
    """

    def __init__(self, orbit_size=90, angle=359.0, interpolation_method="NEAREST",
                 input_dim=None, img_size=None, use_stn=False, **kwargs):
        super().__init__(int(orbit_size), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(360.0, dtype=default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const), name='angle')
        self._orbit_size = orbit_size
        self.use_stn = use_stn

    def orbit_full(self, X):
        img_size = self.img_size(X)
        Ximgs = tf.reshape(X, [-1, img_size, img_size])
        angles = tf.cast(tf.linspace(0., 1., self._orbit_size + 1)[:-1], default_float()) * self.angle
        if self.use_stn:
            return rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
        else:
            return rotate_img_angles(Ximgs, angles, self.interpolation)


ANGLE_JITTER = 1e0  # minimal value for the angle variable (to be safe when transforming to logistic)


class ImageRotation(ImageOrbit):
    def __init__(self, angle=ANGLE_JITTER, interpolation_method="NEAREST", use_stn=False, 
                 input_dim=None, img_size=None, minibatch_size=10, **kwargs):
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(180.0, dtype=default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const))  # constrained to [0, 180]
        self.use_stn = use_stn
        #a=tf.linspace(0.0,1.0,minibatch_size+1)
        #self.b=tf.cast(a, dtype=default_float())

    def orbit_minibatch(self, X):
        # Reparameterise angle
        eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=default_float())
        #eps2 = tf.random.uniform([self.minibatch_size], 0., 1./self.minibatch_size, dtype=default_float())
        #eps=tf.clip_by_value(tf.add(self.b[:self.minibatch_size],eps2),0,1)
        angles = -self.angle + 2. * self.angle * eps
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        
        if self.use_stn:
            return rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
        else:
            return rotate_img_angles(Ximgs, angles, self.interpolation)
        

class GeneralSpatialTransform(ImageOrbit):
    """
    Kernel invariant to transformations using Spatial Transformer Networks (STNs); this correponds to six-parameter
    affine transformations.
    This version of the kernel is parameterised by the six independent parameters directly (thus "_general")
    """

    def __init__(self, theta_min=np.array([1., 0., 0., 0., 1., 0.]),
                 theta_max=np.array([1., 0., 0., 0., 1., 0.]), constrain=False, input_dim=None, img_size=None,
                 minibatch_size=10, initialization=0., colour=False, **kwargs):
        """
        :param theta_min: one end of the range; identity = [1, 0, 0, 0, 1, 0]
        :param theta_max: other end of the range; identity = [1, 0, 0, 0, 1, 0]
        :param constrain: whether theta_min is always below the identity and theta_max always above
        """
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.constrain = constrain
        self.colour = colour
        if constrain:
            self.theta_min_0 = gpflow.Parameter(1. - theta_min[0], dtype=default_float(), transform=positive())
            self.theta_min_1 = gpflow.Parameter(-theta_min[1], dtype=default_float(), transform=positive())
            self.theta_min_2 = gpflow.Parameter(-theta_min[2], dtype=default_float(), transform=positive())
            self.theta_min_3 = gpflow.Parameter(-theta_min[3], dtype=default_float(), transform=positive())
            self.theta_min_4 = gpflow.Parameter(1. - theta_min[4], dtype=default_float(), transform=positive())
            self.theta_min_5 = gpflow.Parameter(-theta_min[5], dtype=default_float(), transform=positive())

            self.theta_max_0 = gpflow.Parameter(theta_min[0], dtype=default_float(), transform=positive(lower=1.))
            self.theta_max_1 = gpflow.Parameter(theta_min[1], dtype=default_float(), transform=positive())
            self.theta_max_2 = gpflow.Parameter(theta_min[2], dtype=default_float(), transform=positive())
            self.theta_max_3 = gpflow.Parameter(theta_min[3], dtype=default_float(), transform=positive())
            self.theta_max_4 = gpflow.Parameter(theta_min[4], dtype=default_float(), transform=positive(lower=1.))
            self.theta_max_5 = gpflow.Parameter(theta_min[5], dtype=default_float(), transform=positive())
        else:
            self.theta_min = gpflow.Parameter(theta_min - initialization, dtype=default_float())
            self.theta_max = gpflow.Parameter(theta_max + initialization, dtype=default_float())

    def orbit_minibatch(self, X):
        eps = tf.random.uniform([self.minibatch_size, 6], 0., 1., dtype=default_float())
        if self.constrain:
            theta_min = tf.stack([1. - self.theta_min_0, -self.theta_min_1, -self.theta_min_2, -self.theta_min_3,
                                  1. - self.theta_min_4, -self.theta_min_5])
            theta_max = tf.stack([self.theta_max_0, self.theta_max_1, self.theta_max_2, self.theta_max_3,
                                  self.theta_max_4, self.theta_max_5])
            theta_min = tf.reshape(theta_min, [1, -1])
            theta_max = tf.reshape(theta_max, [1, -1])
        else:
            theta_min = tf.reshape(self.theta_min, [1, -1])
            theta_max = tf.reshape(self.theta_max, [1, -1])
        thetas = theta_min + (theta_max - theta_min) * eps

        if self.colour:
            return apply_stn_batch_colour(X, thetas)
        else:
            Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
            return apply_stn_batch(Ximgs, thetas)


class InterpretableSpatialTransform(ImageOrbit):
    """
    Kernel invariant to to transformations using Spatial Transformer Networks (STNs); this correponds to six-parameter
    affine transformations.
    This version of the kernel is parameterised by physical components rotation angle, scale in x/y direction, shear in x/y direction: [angle_deg, sx, sy, tx, ty].
    """

    def __init__(self, theta_min=np.array([0., 1., 1., 0., 0.]),
                 theta_max=np.array([0., 1., 1., 0., 0.]), constrain=False, input_dim=None, img_size=None,
                 minibatch_size=10, radians=False, colour=False, **kwargs):
        """
        :param theta_min: one end of the range
        :param theta_max: other end of the range
        :param constrain: whether theta_min is always below the identity and theta_max always above
        """
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.constrain = constrain
        self.radians = radians
        self.colour = colour
        if constrain:
            raise NotImplementedError  # we might want to implement this at some point
        else:
            self.theta_min = gpflow.Parameter(theta_min, dtype=default_float())
            self.theta_max = gpflow.Parameter(theta_max, dtype=default_float())

    def orbit_minibatch(self, X):
        eps = tf.random.uniform([self.minibatch_size, 5], 0., 1., dtype=default_float())
        # only unconstrained version for now
        theta_min = tf.reshape(self.theta_min, [1, -1])
        theta_max = tf.reshape(self.theta_max, [1, -1])
        thetas = theta_min + (theta_max - theta_min) * eps
        stn_thetas = tf.map_fn(lambda thetas: _stn_theta_vec(thetas, radians=self.radians), thetas)

        if self.colour:
            return apply_stn_batch_colour(X, stn_thetas)
        else:
            Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
            return apply_stn_batch(Ximgs, stn_thetas)


class ColorTransform(ImageOrbit):
    """
    Differetiable contrast and brightness adjustment
    """

    def __init__(self, input_dim=None, img_size=None,
                 minibatch_size=10, log_lims_contrast=[-2., 2.], log_lims_brightness=[-2., 2.], **kwargs):
        """
        :param log_lims_contrast: lower and upper end of range for contrast, logit
        :param log_lims_contrast: lower and upper end of range for contrast, logit
        """
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.log_lims_brightness = gpflow.Parameter(log_lims_brightness, dtype=default_float())
        self.log_lims_contrast = gpflow.Parameter(log_lims_contrast, dtype=default_float())

    @property
    def lims_contrast(self):
        return tf.math.sigmoid(self.log_lims_contrast) * 2 - 1

    @property
    def lims_brightness(self):
        return tf.math.sigmoid(self.log_lims_brightness) * 2 - 1

    def orbit_minibatch(self, X):
        # expand X across orbit_size dim.
        X_orbit = tf.tile(tf.expand_dims(X, 1), [1, self.minibatch_size, 1, 1, 1])

        # apply contrast transform first
        # sample contrast_change via reparam. trick
        eps = tf.random.uniform([1, self.minibatch_size, 1, 1, 1], 0., 1., dtype=default_float())
        contrast_change = eps * (self.lims_contrast[1] - self.lims_contrast[0]) + self.lims_contrast[0]
        contrast_change = contrast_change * 255.
        factor = (259. * (contrast_change + 255.)) / (255.* (259. - contrast_change))
        # apply contrast_change and clip values to [0, 1]
        X_orbit = tf.clip_by_value(factor * (X_orbit * 255. - 128.) + 128., 0., 255.) / 255.

        # then apply brightness transform
        eps = tf.random.uniform([1, self.minibatch_size, 1, 1, 1], 0., 1., dtype=default_float())
        brightness_change = eps * (self.lims_brightness[1] - self.lims_brightness[0]) + self.lims_brightness[0]
        X_orbit = tf.clip_by_value(X_orbit + brightness_change, 0., 1.)

        return X_orbit