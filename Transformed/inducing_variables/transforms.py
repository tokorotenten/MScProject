import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import gpflow
from gpflow.config import default_float
from gpflow.utilities.bijectors import positive
from ..kernels.image_transforms import rotate_img_angles, rotate_img_angles_stn, apply_stn_batch, _stn_theta_vec, \
    apply_stn_batch_colour
from ..kernels.transformer import spatial_transformer_network as stn
from ..kernels.orbits import ImageRotation

class Transform(gpflow.base.Module):
    def __init__(self, orbit_size, minibatch_size=None, name=None):
        super().__init__(name=name)
        self._orbit_size = orbit_size
        self.minibatch_size = minibatch_size if minibatch_size is not None else orbit_size

    @property
    def orbit_size(self):
        return self._orbit_size

    def transform_full(self, X):
        raise NotImplementedError

    def transform_minibatch(self, X):
        full_orbit = tf.transpose(self.orbit_full(X), [1, 0, 2])  # [orbit_size, X.shape[0], ...]
        return tf.transpose(tf.random.shuffle(full_orbit)[:self.minibatch_size, :, :], [1, 0, 2])

    def __call__(self, X):
        if self.minibatch_size == self.orbit_size:
            return self.transform_full(X)
        else:
            return self.transform_minibatch(X)
    
    
class ImageOrbit(Transform):
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
        
    def data_size(self,X):
        return tf.shape(X)[0]
        
        
class Mix90(ImageOrbit):
    """
    ImageRot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(4, input_dim=input_dim, img_size=img_size, **kwargs)

    def transform_full(self, X):
        #a=(X.shape[0]*(8/5))/2
        a=40
        #a=200
        X1=X[0:int(a),:]
        X2=X[int(a):,:]
        Ximgs = tf.reshape(X2, [-1, self.img_size(X), self.img_size(X)])
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim(X)))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, self.input_dim(X)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim(X)))
        
        X_new=tf.concat((X2[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)
        X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
        X_return=tf.concat([X1,X_trans],0)
        
        return X_return
    
    
class MixImageRot(ImageOrbit):
    def __init__(self, orbit_size=4, angle=1.0, interpolation_method="NEAREST", use_stn=False,
                 input_dim=None, img_size=None, **kwargs):
        super().__init__(int(orbit_size), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        self._orbit_size = orbit_size
        self.use_stn = use_stn
        self.angle = gpflow.Parameter(angle, trainable=False) 

    def transform_full(self, X):
        # Reparameterise angle
        #eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=default_float())
        #angles = -self.angle + 2. * self.angle * eps
 
        a=200
        X1=X[0:int(a),:]
        X2=X[int(a):,:]
        
        angles = tf.cast(tf.linspace(-1., 1., self._orbit_size + 1)[:-1], default_float()) * self.angle
        Ximgs = tf.reshape(X2, [-1, self.img_size(X), self.img_size(X)])
        if self.use_stn:
            X_new=rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            X_return=tf.concat([X1,X_trans],0)
            return X_return
        else:
            X_new=rotate_img_angles(Ximgs, angles,self.interpolation)
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            X_return=tf.concat([X1,X_trans],0)
            return X_return
    
class Rot90(ImageOrbit):
    """
    ImageRot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(4, input_dim=input_dim, img_size=img_size, **kwargs)

    def transform_full(self, X):
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim(X)))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, self.input_dim(X)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim(X)))
        
        X_new=tf.concat((X[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)
        X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
        
        return X_trans
    
    
class Reflect(ImageOrbit):
    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(2, input_dim=input_dim, img_size=img_size, **kwargs)

    def transform_full(self, X):
        X_switch = -X
        X_new=tf.concat([X[:, None, :], X_switch[:, None, :]], axis=1)
        X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
        return X_trans
    
        
class RotQuant(ImageOrbit):
    """
    ImageRotQuant
    Kernel invariant to any quantised rotations of the input image.
    """

    def __init__(self, orbit_size=4, angle=359.0, interpolation_method="NEAREST",
                 input_dim=None, img_size=None, use_stn=False, **kwargs):
        super().__init__(int(orbit_size), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(360.0, dtype=default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const), name='angle')
        self._orbit_size = orbit_size
        self.use_stn = use_stn

    def transform_full(self, X):
        img_size = self.img_size(X)
        Ximgs = tf.reshape(X, [-1, img_size, img_size])
        angles = tf.cast(tf.linspace(0., 1., self._orbit_size + 1)[:-1], default_float()) * self.angle
        if self.use_stn:
            X_new=rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans
        else:
            X_new=rotate_img_angles(Ximgs, angles,self.interpolation)
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans      
    

class ImageRot(ImageOrbit):
    def __init__(self, orbit_size=4, angle=1.0, interpolation_method="NEAREST", use_stn=False,
                 input_dim=None, img_size=None, **kwargs):
        super().__init__(int(orbit_size), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        self._orbit_size = orbit_size
        self.use_stn = use_stn
        self.angle = gpflow.Parameter(angle, trainable=False) 

    def transform_full(self, X):
        # Reparameterise angle
        #eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=default_float())
        #angles = -self.angle + 2. * self.angle * eps
 
        #angles = tf.cast(tf.linspace(-1., 1., self._orbit_size + 1)[:-1], default_float()) * self.angle
        angles = tf.cast(tf.linspace(-1., 1., self._orbit_size), default_float()) * self.angle
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        if self.use_stn:
            X_new=rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans
        else:
            X_new=rotate_img_angles(Ximgs, angles,self.interpolation)
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans     
    
    
class ImageRot0(ImageOrbit):
    def __init__(self, orbit_size=4, angle=1.0, interpolation_method="NEAREST", use_stn=False,
                 input_dim=None, img_size=None, **kwargs):
        super().__init__(int(orbit_size), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(180.0, dtype=default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const))  # constrained to [0, 180]
        self._orbit_size = orbit_size
        self.use_stn = use_stn

    def transform_full(self, X):
        # Reparameterise angle
        #eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=default_float())
        #angles = -self.angle + 2. * self.angle * eps
        angles = tf.cast(tf.linspace(-1., 1., self._orbit_size + 1)[:-1], default_float()) * self.angle
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        if self.use_stn:
            X_new=rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans
        else:
            X_new=rotate_img_angles(Ximgs, angles,self.interpolation)
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans 
        
        
class ImageRot2(ImageOrbit):
    def __init__(self, theta_init=np.array([0.1, 90.0, 180.0, 359.9]), interpolation_method="NEAREST", use_stn=True,
                 input_dim=None, img_size=None, **kwargs):
        super().__init__(int(4), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(360.0, dtype=default_float())
        
        self.theta1=gpflow.Parameter(theta_init[0], transform=tfb.Sigmoid(low_const, high_const))
        self.theta2=gpflow.Parameter(theta_init[1], transform=tfb.Sigmoid(low_const, high_const))
        self.theta3=gpflow.Parameter(theta_init[2], transform=tfb.Sigmoid(low_const, high_const))
        self.theta4=gpflow.Parameter(theta_init[3], transform=tfb.Sigmoid(low_const, high_const))
        
        self.use_stn = use_stn

    def transform_full(self, X):
        # Reparameterise angle
        #eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=default_float())
        #angles = -self.angle + 2. * self.angle * eps
        angles=tf.stack([self.theta1,self.theta2,self.theta3,self.theta4])
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        if self.use_stn:
            X_new=rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans
        else:
            X_new=rotate_img_angles(Ximgs, angles,self.interpolation)
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans 
        

class Rot902(ImageOrbit):
    """
    ImageRot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(2, input_dim=input_dim, img_size=img_size, **kwargs)
        dice=list(range(1,4))
        a=np.zeros((1000,3))
        for i in range(len(a)):
            samples=np.random.choice(dice)
            a[i,samples-1]=1
        b=np.ones((1000,1))
        a2=np.hstack([b,a])
        a3=a2.reshape(-1,)
        self.ind=np.where(a3==1)[0]

    def transform_full(self, X):
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        indx=self.ind[:np.shape(X)[0]*2]
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim(X)))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, self.input_dim(X)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim(X)))
        
        X_new=tf.concat((X[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)
        X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
        
        return tf.gather(X_trans,indx)
    
class Rot903(ImageOrbit):
    """
    ImageRot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(1, input_dim=input_dim, img_size=img_size, **kwargs)
        dice=list(range(1,5))
        a=np.zeros((1000,4))
        for i in range(len(a)):
            samples=np.random.choice(dice)
            a[i,samples-1]=1
        a2=a.reshape(-1,)
        self.ind=np.where(a2==1)[0]

    def transform_full(self, X):
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        indx=self.ind[:np.shape(X)[0]]
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim(X)))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, self.input_dim(X)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim(X)))
        
        X_new=tf.concat((X[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)
        X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
        
        return tf.gather(X_trans,indx)
    
    
    
class RotQuant2(ImageOrbit):
    """
    ImageRotQuant
    Kernel invariant to any quantised rotations of the input image.
    """

    def __init__(self, orbit_size=4, angle=359.0, interpolation_method="NEAREST",
                 input_dim=None, img_size=None, use_stn=False, **kwargs):
        super().__init__(int(orbit_size), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(360.0, dtype=default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const), name='angle')
        self._orbit_size = orbit_size
        self.use_stn = use_stn
        
        dice=list(range(1,orbit_size+1))
        a=np.zeros((1000,orbit_size))
        for i in range(len(a)):
            samples=np.random.choice(dice)
            a[i,samples-1]=1
        a2=a.reshape(-1,)
        self.ind=np.where(a2==1)[0]

    def transform_full(self, X):
        img_size = self.img_size(X)
        Ximgs = tf.reshape(X, [-1, img_size, img_size])
        indx=self.ind[:np.shape(X)[0]]
        angles = tf.cast(tf.linspace(0., 1., self._orbit_size + 1)[:-1], default_float()) * self.angle
        if self.use_stn:
            X_new=rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return X_trans
        else:
            X_new=rotate_img_angles(Ximgs, angles,self.interpolation)
            X_trans=tf.reshape(X_new,[-1,X_new.shape[2]])
            return tf.gather(X_trans,indx)