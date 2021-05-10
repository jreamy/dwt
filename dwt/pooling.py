
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.python.keras.utils import conv_utils

from .wavelet import Wavelet


class DWTPooling(L.Layer, Wavelet):
    """Abstract N-D Discrete Wavelet Transform layer used as implementation base.

    This layer applies the Discrete Wavelet Transform across the channels of a
    given batch of data.

    Arguments:
        wavelet: A string, dwt.Wavelet or pywt.Wavelet, the wavelet or name of the
            wavelet to use in the layer.
        level: The level of refinement of the filter, a higher level has a lower
            cutoff frequency, but contains more of the filter features.
        strides: An integer or tuple/list of n integers, specifying the stride
            length of the pooling layer. The Discrete Wavelet Transform uses 2.
        rank: An integer, the rank of the input data, e.g. "2" for 2D image, etc.
        mode: One of `"symmetric"`, `"constant"`, `"reflect"`, (case-insensitive).
            The method of padding the input data to use. Used as input to `tf.pad`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds
            to inputs with shape `(batch_size, ..., channels)` while `channels_first`
            corresponds to inputs with shape `(batch_size, channels, ...)`.
        **kwargs: keyword arguments passed to the Layer initializer.
    """

    def __init__(
        self,
        wavelet,
        level=1,
        strides=2,
        rank=1,
        mode='symmetric',
        data_format=None,
        **kwargs
    ):
        L.Layer.__init__(self, **kwargs)
        Wavelet.__init__(self, wavelet, level)

        self.mode = mode
        self.rank = rank

        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.data_format = conv_utils.normalize_data_format(data_format)
        self._channels_first = self.data_format == "channels_first"

        # All the convolutions will be permuted to channels last
        self.conv_format = conv_utils.convert_data_format(
            "channels_last", self.rank+2)

        self.filters = self.build_filters(rank, self.dec_lo, self.dec_hi)

        offset = 1 if self.p % 2 or self.p <= 2 else 2
        padding = self.rank * ([self.p-1, self.p-offset],)

        if self._channels_first:
            self.padding = tf.constant([[0, 0], [0, 0], *padding])
        else:
            self.padding = tf.constant([[0, 0], *padding, [0, 0]])

    def call(self, x):
        return [self.conv(x, f) for f in self.filters]

    def conv(self, x, kernel):

        x = tf.pad(x, self.padding, mode=self.mode)

        if not self._channels_first:
            x = K.permute_dimensions(
                x, (0, self.rank+1, *range(1, self.rank+1)))

        # These keras bits were easier to work with than manually reshaping
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

        if not self._channels_first:
            x = K.permute_dimensions(x, (0, *range(2, self.rank+2), 1))

        return x

    def compute_output_shape(self, x):
        if self._channels_first:
            inner = [d//self.strides for d in x[2:]]
            return len(self.filters) * (x[0], x[1], *inner)

        inner = [d//self.strides for d in x[1:-1]]
        return len(self.filters) * (x[0], *inner, x[-1])

    def get_config(self):
        config = {
            "wavelet": self.wavelet.name,
            "level": self.wavelet.level,
            "strides": self.strides,
            "rank": self.rank,
            "mode": self.mode,
            "data_format": self.data_format,
        }

        base_config = L.Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))


class DWTPooling1D(DWTPooling):
    """1D Discrete Wavelet Transform (e.g. temporal DWT).

    Returns 2 tensors, with low and high pass filtering applied
    """

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
            rank=1,
            strides=strides,
            mode=mode,
            data_format=data_format,
            **kwargs
        )


class DWTPooling2D(DWTPooling):
    """2D Discrete Wavelet Transform (e.g. spatial DWT over image).

    Returns 4 tensors with lowpass, horizontal, vertical, and diagonal
    filtering applied.
    """

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
            rank=2,
            strides=strides,
            mode=mode,
            data_format=data_format,
            **kwargs
        )


class DWTPooling3D(DWTPooling):
    """3D Discrete Wavelet Transform (e.g. spatial DWT over volumes).

    Returns 8 tensors with LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH filtering.
    """

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
            rank=3,
            strides=strides,
            mode=mode,
            data_format=data_format,
            **kwargs
        )
