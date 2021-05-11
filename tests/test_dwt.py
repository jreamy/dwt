
import unittest
import itertools as it
import numpy as np

from tensorflow.keras import layers as L
from tensorflow.keras import models as M

from context import dwt
from helpers import *

batch = 3
channels = 5

wavelet_count = 10 if run_short else None


class TestDWTPooling1D(unittest.TestCase):

    def test_orthogonal_wavelets(self):

        wavelets = get_wavelets(count=wavelet_count)
        lengths = fill_similar(
            [1017, 1023, 1024, 1025, 1037], len(wavelets))

        for (wave, length) in zip(wavelets, lengths):
            with self.subTest(wave=wave.name, length=length):
                self.channels_last_test(wave, length)

            with self.subTest(wave=wave.name, length=length):
                self.channels_first_test(wave, length)

    def channels_last_test(self, wave, length):
        print(wave)

        signal = np.random.random((batch, length, channels))

        # Build a simple model
        inp = L.Input(signal.shape[1:])
        x = dwt.DWTPooling1D(wave)(inp)
        model = M.Model(inp, x)
        model.summary()

        # Get the model and pywt output
        dwt_out = model(signal)
        pwt_out = pywt.dwt(signal, wave, axis=1)

        # Check per-tensor similarity
        for (d_out, p_out) in zip(dwt_out, pwt_out):
            print(d_out[0, :10, 0])
            print(p_out[0, :10, 0])

            # Assert the correct output shape was achieved
            self.assertEqual(d_out.shape, (batch, length//2, channels))

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :length//2, :]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))

    def channels_first_test(self, wave, length):
        print(wave)

        signal = np.random.random((batch, channels, length))

        # Build a simple model
        inp = L.Input(signal.shape[1:])
        x = dwt.DWTPooling1D(wave, data_format='channels_first')(inp)
        model = M.Model(inp, x)
        model.summary()

        # Get the model and pywt output
        dwt_out = model(signal)
        pwt_out = pywt.dwt(signal, wave, axis=2)

        # Check per-tensor similarity
        for (d_out, p_out) in zip(dwt_out, pwt_out):
            print(d_out[0, 0, :10])
            print(p_out[0, 0, :10])

            # Assert the correct output shape was achieved
            self.assertEqual(d_out.shape, (batch, channels, length//2))

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :, :length//2]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))


if __name__ == '__main__':
    unittest.main()
