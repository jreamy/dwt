import pywt
import numpy as np
import random
import itertools as it
import pytest
import math

from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import models as M

from .context import dwt

wavelets = [
    pywt.Wavelet(x) for x in pywt.wavelist(
        kind="discrete") if pywt.Wavelet(x).orthogonal
]


@pytest.mark.dim1
@pytest.mark.parametrize("wave,length", it.product(
    random.sample(wavelets, 5), (1023, 1024, 1025)
))
def test_pooling_1d_orthogonal(wave, length):
    print(wave)

    signal = np.random.random((3, length, 5))

    # Build a simple model
    inp = L.Input(signal.shape[1:])
    x = dwt.DWTPooling1D(wave)(inp)
    model = M.Model(inp, x)
    model.summary()

    dwt_out = model(signal)
    pwt_out = pywt.dwt(signal, wave, axis=1)

    for (d_out, p_out) in zip(dwt_out, pwt_out):
        print(d_out[0, :10, 0])
        print(p_out[0, :10, 0])

        assert d_out.shape == (3, length//2, 5)

        d_out = d_out.numpy()
        p_out = p_out[:, :length//2, :]

        # check approximate equality via mse
        assert np.square(d_out - p_out).mean() == pytest.approx(0)


@pytest.mark.dim2
@pytest.mark.parametrize("wave,length,channels", it.product(
    random.sample([x for x in wavelets if x.dec_len < 20], 5),
    (63, 64, 65), ("first", "last"),
))
def test_pooling_2d_orthogonal(wave, length, channels):
    print(wave)

    if channels == "last":
        signal = np.random.random((3, length, length, 5))

        pwt_out = pywt.dwt2(signal, wave, axes=(1, 2))
        pwt_out = (pwt_out[0], *pwt_out[1])
        pwt_out = [p[:, :length//2, :length//2, :] for p in pwt_out]
    else:
        signal = np.random.random((3, 5, length, length))

        pwt_out = pywt.dwt2(signal, wave, axes=(2, 3))
        pwt_out = (pwt_out[0], *pwt_out[1])
        pwt_out = [p[:, :, :length//2, :length//2] for p in pwt_out]

    # Build a simple model
    inp = L.Input(signal.shape[1:])
    x = dwt.DWTPooling2D(wave, data_format="channels_"+channels)(inp)
    model = M.Model(inp, x)
    model.summary()

    dwt_out = model(signal)

    for (d_out, p_out) in zip(dwt_out, pwt_out):
        print(d_out[0, 0, :10, 0])
        print(p_out[0, 0, :10, 0])

        assert d_out.shape == p_out.shape
        d_out = d_out.numpy()

        # check approximate equality via mse
        assert np.square(d_out - p_out).mean() == pytest.approx(0)


@pytest.mark.dim3
@pytest.mark.parametrize("wave,length", it.product(
    [x for x in wavelets if x.dec_len < 6], (7, 8, 9)
))
def test_pooling_3d_orthogonal(wave, length):
    print(wave)

    signal = np.random.random((3, length, length, length, 5))

    # Build a simple model
    inp = L.Input(signal.shape[1:])
    x = dwt.DWTPooling3D(wave)(inp)
    model = M.Model(inp, x)
    model.summary()

    dwt_out = model(signal)
    pwt_out = pywt.dwtn(signal, wave, axes=(3, 2, 1))
    pwt_out = [pwt_out[k] for k in sorted(pwt_out.keys())]

    for (d_out, p_out) in zip(dwt_out, pwt_out):
        print(d_out[0, :10, 0, 0, 0])
        print(p_out[0, :10, 0, 0, 0])

        l = length//2
        assert d_out.shape == (3, l, l, l, 5)

        d_out = d_out.numpy()
        p_out = p_out[:, :l, :l, :l, :]

        # check approximate equality via mse
        assert np.square(d_out - p_out).mean() == pytest.approx(0)
