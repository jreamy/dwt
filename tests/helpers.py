
import os
import random
import unittest

import pywt
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


# Environment variables used to specify test setup
run_short = bool(os.environ.get("SHORT"))


def get_wavelets(kind="discrete", orthogonality="orthogonal", count=None, max_length=None):
    wavelets = [pywt.Wavelet(x) for x in pywt.wavelist(kind=kind)]

    if orthogonality == "orthogonal":
        wavelets = [x for x in wavelets if x.orthogonal]
    elif orthogonality == "biorthogonal":
        wavelets = [x for x in wavelets if x.biorthogonal]

    if max_length:
        wavelets = [x for x in wavelets if x.dec_len <= max_length]

    if count and count < len(wavelets):
        wavelets = random.sample(wavelets, count)

    return wavelets


def fill_similar(data, length):
    if length <= len(data):
        return random.sample(data, length)

    mn, mx = min(data), max(data)

    return data + [random.randint(mn, mx) for _ in range(length-len(data))]


def show_data(data):
    data = tuple(data)
    fmt = " ".join(["%0.5f"] * len(data))
    print(fmt % data)
