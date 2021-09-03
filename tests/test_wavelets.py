
import unittest
import itertools as it
import numpy as np

from context import dwt
from helpers import *

wavelet_count = 10 if run_short else None


class Test_Wavelets(unittest.TestCase):

    def test_orthogonal_wavelets(self):
        wavelets = get_wavelets(count=wavelet_count)
        levels = fill_similar([1, 4, 5, 8], len(wavelets))

        for (wave, level) in zip(wavelets, levels):
            with self.subTest(wave=wave.name, level=level):
                self.wavelet_filter_test(wave.name, level)

    def wavelet_filter_test(self, wave, level):
        w = dwt.Wavelet(wave, level)
        p = pywt.Wavelet(wave).wavefun(level)

        filters = [(w.dec_lo, p[0]), (w.dec_hi, p[1])]
        if not w.wavelet.biorthogonal:
            filters.extend([(w.rec_lo, p[2]), (w.rec_hi, p[3])])

        for dwt_f, pwt_f in filters:
            dwt_f = np.reshape(dwt_f, (-1,))
            pwt_f = np.reshape(pwt_f, (-1,))

            self.assertTrue(np.allclose(dwt_f, pwt_f))
