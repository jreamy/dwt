
import pywt
import numpy as np


class Wavelet:

    def __init__(self, wavelet, level=1):
        self.wavelet = wavelet
        if not isinstance(wavelet, pywt.Wavelet):
            self.wavelet = pywt.Wavelet(self.wavelet)

        self.level = level

        if self.wavelet.orthogonal:
            print("orthog")
            self.dec_lo, self.dec_hi, _ = self.wavelet.wavefun(self.level)
            self.rec_lo, self.rec_hi = self.dec_lo, self.dec_hi

        elif self.wavelet.biorthogonal:
            print("bithog")
            self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi, _ = self.wavelet.wavefun(
                self.level)

        else:
            raise ValueError("wavelet not supported: %s" % wavelet)

        self.p = self.wavelet.dec_len
