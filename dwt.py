
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import layers as L

from tensorflow.python.keras.utils import conv_utils

from wavelet import Wavelet


class DWTPooling(L.Layer, Wavelet):
    def __init__(
        self,
        wavelet,
        level=1,
        strides=2,
        ndim=1,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):
        L.Layer.__init__(self, **kwargs)
        Wavelet.__init__(self, wavelet, level)

        self.strides = strides
        self.mode = mode
        self.data_format = conv_utils.normalize_data_format(data_format)

        self.ndim = ndim
        self.mag = np.power(2, self.ndim * self.level / 2)

        filters = [self.dec_lo, self.dec_hi]
        for _ in range(1, ndim):
            dec_lo = [self.kron(f, self.dec_lo) for f in filters]
            dec_hi = [self.kron(f, self.dec_hi) for f in filters]

            filters = [f for pair in [dec_lo, dec_hi] for f in pair]

        self.filters = [self.build_kernel(f) for f in filters]

        padding = self.ndim * ([self.p, self.p],)
        self.padding = tf.constant([[0, 0], [0, 0], *padding])

        if self.ndim == 1:
            self.conv_format = "NWC"
        elif self.ndim == 2:
            self.conv_format = "NHWC"
        elif self.ndim == 3:
            self.conv_format = "NDHWC"

    def kron(self, kernel, filter):
        sh = (*kernel.shape, len(filter))
        return np.kron(kernel, filter).reshape(sh)

    def call(self, x):
        return [self.conv(x, f) for f in self.filters]

    def compute_output_shape(self, x):
        if self.data_format == 'channels_first':
            inner = [np.ceil(np.ceil(d/self.strides) for d in x[2:])]
            return len(self.filters) * (x[0], x[1], *inner)

        inner = [np.ceil(np.ceil(d/self.strides) for d in x[1:-1])]
        return len(self.filters) * (x[0], *inner, x[-1])

    def conv(self, x, kernel):

        if self.data_format == 'channels_last':
            x = K.permute_dimensions(
                x, (0, self.ndim+1, *range(1, self.ndim+1)))

        x = tf.pad(x, self.padding, mode=self.mode)

        x = K.expand_dims(x, axis=-1)
        x = L.TimeDistributed(
            L.Lambda(lambda x: tf.nn.convolution(
                x, kernel, padding="SAME", data_format=self.conv_format))
        )(x)
        x = K.squeeze(x, axis=-1)

        if self.ndim == 1:
            x = x[:, :, self.p:-self.p:self.strides]
        elif self.ndim == 2:
            x = x[:, :, self.p:-self.p:self.strides,
                  self.p:-self.p:self.strides]
        elif self.ndim == 3:
            x = x[:, :, self.p:-self.p:self.strides, self.p:-
                  self.p:self.strides, self.p:-self.p:self.strides]

        if self.data_format == 'channels_last':
            x = K.permute_dimensions(x, (0, *range(2, self.ndim+2), 1))

        return x

    def build_kernel(self, kernel):
        kernel = np.reshape(kernel, (*kernel.shape, 1, 1)) / self.mag
        return tf.convert_to_tensor(kernel, dtype=self.dtype)


class DWTPooling1D(DWTPooling):

    def __init__(
        self,
        wavelet,
        level=1,
        strides=2,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):

        super(DWTPooling1D, self).__init__(
            wavelet,
            level=level,
            ndim=1,
            strides=strides,
            mode=mode,
            data_format=data_format,
            **kwargs
        )


class DWTPooling2D(DWTPooling):

    def __init__(
        self,
        wavelet,
        level=1,
        strides=2,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):

        super(DWTPooling2D, self).__init__(
            wavelet,
            level=level,
            ndim=2,
            strides=strides,
            mode=mode,
            data_format=data_format,
            **kwargs
        )


class DWTPooling3D(DWTPooling):

    def __init__(
        self,
        wavelet,
        level=1,
        strides=2,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):

        super(DWTPooling3D, self).__init__(
            wavelet,
            level=level,
            ndim=3,
            strides=strides,
            mode=mode,
            data_format=data_format,
            **kwargs
        )
