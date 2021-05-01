
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
        self.filters = self.build_filters(ndim, self.dec_lo, self.dec_hi)

        offset = 1 if self.p % 2 or self.p <= 2 else 2
        padding = self.ndim * ([self.p-1, self.p-offset],)
        self.padding = tf.constant([[0, 0], [0, 0], *padding])

        if self.ndim == 1:
            self.conv_format = "NWC"
        elif self.ndim == 2:
            self.conv_format = "NHWC"
        elif self.ndim == 3:
            self.conv_format = "NDHWC"

    def call(self, x):
        return [self.conv(x, f) for f in self.filters]

    def compute_output_shape(self, x):
        if self.data_format == 'channels_first':
            inner = [d//self.strides for d in x[2:]]
            return len(self.filters) * (x[0], x[1], *inner)

        inner = [d//self.strides for d in x[2:]]
        return len(self.filters) * (x[0], *inner, x[-1])

    def conv(self, x, kernel):

        if self.data_format == 'channels_last':
            x = K.permute_dimensions(
                x, (0, self.ndim+1, *range(1, self.ndim+1)))

        x = tf.pad(x, self.padding, mode=self.mode)

        x = K.expand_dims(x, axis=-1)
        x = L.TimeDistributed(
            L.Lambda(lambda x: tf.nn.convolution(
                x, kernel,
                padding="VALID",
                strides=self.strides,
                data_format=self.conv_format,
            ))
        )(x)
        x = K.squeeze(x, axis=-1)

        if self.data_format == 'channels_last':
            x = K.permute_dimensions(x, (0, *range(2, self.ndim+2), 1))

        return x


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
