
import pywt
import numpy as np
import itertools as it
import pytest

from .context import dwt


@pytest.mark.parametrize("wave,level", it.product(pywt.wavelist(kind="discrete"), (1, 5, 9)))
def test_wavelet_filters(wave, level):

    # Get the custom and pywt wavelets
    w = dwt.Wavelet(wave, level)
    p = pywt.Wavelet(wave).wavefun(level)

    # Get the custom and pywt filters
    filters = [(w.dec_lo, p[0]), (w.dec_hi, p[1])]
    if not w.wavelet.biorthogonal:
        filters.extend([(w.rec_lo, p[2]), (w.rec_hi, p[3])])

    # Assert filter equality
    for tf_filter, pw_filter in filters:
        tf_filter = np.reshape(tf_filter, (-1,))
        pw_filter = np.reshape(pw_filter, (-1,))

        assert tf_filter == pytest.approx(pw_filter)
