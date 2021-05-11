
import os
import random
import unittest
from argparse import ArgumentParser

from contextlib import redirect_stdout
from io import StringIO

import pywt

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
