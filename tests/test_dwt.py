
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


class Test_DWTPooling1D(unittest.TestCase):

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
            show_data(d_out[0, :5, 0])
            show_data(p_out[0, :5, 0])

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
            show_data(d_out[0, 0, :5])
            show_data(p_out[0, 0, :5])

            # Assert the correct output shape was achieved
            self.assertEqual(d_out.shape, (batch, channels, length//2))

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :, :length//2]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))


class Test_DWTPooling2D(unittest.TestCase):

    def test_orthogonal_wavelets(self):

        wavelets = get_wavelets(count=wavelet_count, max_length=20)
        widths = fill_similar(
            [57, 63, 64, 65, 69], len(wavelets))
        heights = fill_similar(
            [69, 65, 64, 63, 57], len(wavelets))

        for (wave, width, height) in zip(wavelets, widths, heights):
            with self.subTest(wave=wave.name, width=width, height=height):
                self.channels_last_test(wave, width, height)

            with self.subTest(wave=wave.name, width=width, height=height):
                self.channels_first_test(wave, width, height)

    def channels_last_test(self, wave, width, height):
        print(wave)

        signal = np.random.random((batch, width, height, channels))

        # Build a simple model
        inp = L.Input(signal.shape[1:])
        x = dwt.DWTPooling2D(wave)(inp)
        model = M.Model(inp, x)
        model.summary()

        # Get the model and pywt output
        dwt_out = model(signal)
        pwt_out = pywt.dwt2(signal, wave, axes=(1, 2))
        pwt_out = (pwt_out[0], *pwt_out[1])

        # Check per-tensor similarity
        for (d_out, p_out) in zip(dwt_out, pwt_out):
            show_data(d_out[0, 0, :5, 0])
            show_data(p_out[0, 0, :5, 0])

            # Assert the correct output shape was achieved
            self.assertEqual(
                d_out.shape, (batch, width//2, height//2, channels)
            )

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :width//2, :height//2, :]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))

    def channels_first_test(self, wave, width, height):
        print(wave)

        signal = np.random.random((batch, channels, width, height))

        # Build a simple model
        inp = L.Input(signal.shape[1:])
        x = dwt.DWTPooling2D(wave, data_format='channels_first')(inp)
        model = M.Model(inp, x)
        model.summary()

        # Get the model and pywt output
        dwt_out = model(signal)
        pwt_out = pywt.dwt2(signal, wave, axes=(2, 3))
        pwt_out = (pwt_out[0], *pwt_out[1])

        # Check per-tensor similarity
        for (d_out, p_out) in zip(dwt_out, pwt_out):
            show_data(d_out[0, 0, :5, 0])
            show_data(p_out[0, 0, :5, 0])

            # Assert the correct output shape was achieved
            self.assertEqual(
                d_out.shape, (batch, channels, width//2, height//2)
            )

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :, :width//2, :height//2]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))


class Test_DWTPooling3D(unittest.TestCase):

    def test_orthogonal_wavelets(self):

        wavelets = get_wavelets(count=wavelet_count, max_length=6)
        lengths = fill_similar([11, 13, 15, 16, 17], len(wavelets))
        widths = fill_similar([11, 13, 15, 16, 17], len(wavelets))
        heights = fill_similar([11, 13, 15, 16, 17], len(wavelets))

        for (wave, length, width, height) in zip(wavelets, lengths, widths, heights):
            with self.subTest(wave=wave.name, length=length, width=width, height=height):
                self.channels_last_test(wave, length, width, height)

            with self.subTest(wave=wave.name, length=length, width=width, height=height):
                self.channels_first_test(wave, length, width, height)

    def channels_last_test(self, wave, length, width, height):
        print(wave)

        signal = np.random.random((batch, length, width, height, channels))

        # Build a simple model
        inp = L.Input(signal.shape[1:])
        x = dwt.DWTPooling3D(wave)(inp)
        model = M.Model(inp, x)
        model.summary()

        # Get the model and pywt output
        dwt_out = model(signal)
        pwt_out = pywt.dwtn(signal, wave, axes=(3, 2, 1))
        pwt_out = [pwt_out[k] for k in sorted(pwt_out.keys())]

        # Check per-tensor similarity
        for (d_out, p_out) in zip(dwt_out, pwt_out):
            show_data(d_out[0, 0, :5, 0, 0])
            show_data(p_out[0, 0, :5, 0, 0])

            # Assert the correct output shape was achieved
            self.assertEqual(
                d_out.shape, (batch, length//2, width//2, height//2, channels)
            )

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :length//2, :width//2, :height//2, :]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))

    def channels_first_test(self, wave, length, width, height):
        print(wave)

        signal = np.random.random((batch, channels, length, width, height))

        # Build a simple model
        inp = L.Input(signal.shape[1:])
        x = dwt.DWTPooling3D(wave, data_format='channels_first')(inp)
        model = M.Model(inp, x)
        model.summary()

        # Get the model and pywt output
        dwt_out = model(signal)
        pwt_out = pywt.dwtn(signal, wave, axes=(4, 3, 2))
        pwt_out = [pwt_out[k] for k in sorted(pwt_out.keys())]

        # Check per-tensor similarity
        for (d_out, p_out) in zip(dwt_out, pwt_out):
            show_data(d_out[0, 0, :5, 0, 0])
            show_data(p_out[0, 0, :5, 0, 0])

            # Assert the correct output shape was achieved
            self.assertEqual(
                d_out.shape, (batch, channels, length//2, width//2, height//2)
            )

            # Trim the last values from the pywt output
            d_out = d_out.numpy()
            p_out = p_out[:, :, :length//2, :width//2, :height//2]

            # Assert the correct output values were achieved
            self.assertTrue(np.allclose(
                d_out, p_out, rtol=0.0001, atol=0.0001
            ))


if __name__ == '__main__':
    unittest.main()
