
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import layers as L

from tensorflow.python.keras.utils import conv_utils

from .wavelet import Wavelet


class DWTUpSampling(L.Layer, Wavelet):
    def __init__(
        self,
        wavelet,
        level=1,
        rank=1,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):
        L.Layer.__init__(self, **kwargs)
        Wavelet.__init__(self, wavelet, level)

        self.mode = mode
        self.rank = rank

        self.data_format = conv_utils.normalize_data_format(data_format)
        self._channels_first = self.data_format == "channels_first"

        # All the convolutions will be permuted to channels last
        self.conv_format = conv_utils.convert_data_format(
            "channels_last", self.rank+2)

        self.filters = self.build_filters(
            self.rank, self.rec_lo[::-1], self.rec_hi[::-1])

        self.offset = 1 if self.p % 2 or self.p <= 2 else 2
        padding = self.rank * ([self.p//2, self.p//2],)

        if self._channels_first:
            self.padding = tf.constant([[0, 0], [0, 0], *padding])
        else:
            self.padding = tf.constant([[0, 0], *padding, [0, 0]])

    def call(self, data):
        return sum([self.conv_transpose(x, f) for (x, f) in zip(data, self.filters)])

    def build(self, input_shapes):
        shapes = set([tuple(s) for s in input_shapes])
        if len(shapes) != 1:
            raise ValueError(
                'requires all inputs to have same shape, got %s' % shapes)

        input_shape = shapes.pop()
        idx = 2 if self._channels_first else 1
        spatial_dims = input_shape[idx: idx+self.rank]

        self.upsamplers = [self._build_upsampling(
            length) for length in spatial_dims]

    def _build_upsampling(self, length):
        # Account for padding
        length += self.p

        # Upsamplers insert zeros between elements
        upsampler = np.zeros((length, 2*length-1))
        for i in range(0, length):
            upsampler[i, 2*i] = 1

        return tf.convert_to_tensor(upsampler, dtype=self.dtype)

    def conv_transpose(self, x, kernel):

        # Pad the ends of the input data
        x = tf.pad(x, self.padding, mode=self.mode)

        # Insert zeros between alll elements
        axis = 2 if self._channels_first else 1
        for upsampler in self.upsamplers:
            x = tf.tensordot(x, upsampler, axes=(axis, 0))

        x = K.expand_dims(x, axis=-1)
        x = L.TimeDistributed(
            L.Lambda(lambda x: tf.nn.convolution(
                x, kernel,
                padding="VALID",
                data_format=self.conv_format,
            ))
        )(x)
        x = K.squeeze(x, axis=-1)

        # This is inelegant, but works for removing the first element if needed
        if self.offset % 2 == 0:
            if self.rank == 1:
                x = x[:, :, 1:]
            elif self.rank == 2:
                x = x[:, :, 1:, 1:]
            elif self.rank == 3:
                x = x[:, :, 1:, 1:, 1:]

        if not self._channels_first:
            x = K.permute_dimensions(x, (0, *range(2, self.rank+2), 1))

        return x


class DWTUpSampling1D(DWTUpSampling):

    def __init__(
        self,
        wavelet,
        level=1,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):

        super(DWTUpSampling1D, self).__init__(
            wavelet,
            level=level,
            rank=1,
            mode=mode,
            data_format=data_format,
            **kwargs
        )


class DWTUpSampling2D(DWTUpSampling):

    def __init__(
        self,
        wavelet,
        level=1,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):

        super(DWTUpSampling2D, self).__init__(
            wavelet,
            level=level,
            rank=2,
            mode=mode,
            data_format=data_format,
            **kwargs
        )


class DWTUpSampling3D(DWTUpSampling):

    def __init__(
        self,
        wavelet,
        level=1,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):

        super(DWTUpSampling3D, self).__init__(
            wavelet,
            level=level,
            rank=3,
            mode=mode,
            data_format=data_format,
            **kwargs
        )
