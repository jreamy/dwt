
import pywt
import numpy as np

import tensorflow as tf


class Wavelet:

    def __init__(self, wavelet, level=1):
        if isinstance(wavelet, Wavelet):
            self.wavelet = wavelet.wavelet
        elif isinstance(wavelet, pywt.Wavelet):
            self.wavelet = wavelet
        else:
            self.wavelet = pywt.Wavelet(wavelet)

        self.level = level
        self.p = self.wavelet.dec_len

        if self.wavelet.orthogonal:
            self.dec_lo, self.dec_hi, _ = self.wavelet.wavefun(self.level)
            self.rec_lo, self.rec_hi = self.dec_lo, self.dec_hi

        elif self.wavelet.biorthogonal:
            self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi, _ = self.wavelet.wavefun(
                self.level)

        else:
            raise ValueError("wavelet not supported: %s" % wavelet)

    def kron(self, kernel, filter):
        sh = (*kernel.shape, len(filter))
        return np.kron(kernel, filter).reshape(sh)

    def build_kernel(self, kernel):
        kernel = np.reshape(kernel, (*kernel.shape, 1, 1))
        return tf.convert_to_tensor(kernel, dtype=self.dtype)

    def build_filters(self, ndim, f_lo, f_hi):
        filters = [f_lo, f_hi]
        mag = np.power(2, ndim * self.level / 2)

        for _ in range(1, ndim):
            dec_lo = [self.kron(f, f_lo) for f in filters]
            dec_hi = [self.kron(f, f_hi) for f in filters]

            filters = [f for pair in [dec_lo, dec_hi] for f in pair]

        return [self.build_kernel(f / mag) for f in filters]
