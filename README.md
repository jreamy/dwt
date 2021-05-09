# Keras Discrete Wavelet Transform Implementations

This project is still in its infancy, but I want to make the discrete wavelet transform accessible in keras machine learning projects. I'm a lone developer right now, and cannot commit to regular releases or implementing all wavelet functions. I intend to build out more features, examples, and tests. I'll also create actual documentation soon, standardize names, and generally make the code less hacky...

## Wavelets

Currently only the orthogonal wavelets work, but I plan on adding support for the biorthogonal wavelets.

Working wavelets include:
 - Haar (haar)
 - Coiflets (coif1-coif17)
 - Daubechies (db1-db38)
 - Symlets (sym2-sym20)
 - Discrete Meyer (dmey)

## Examples

You can create a keras layers similar to other standard layers
```python
from tensorflow.keras import Layers as L
import dwt

inp = L.Input((1024, 3))
cA, cD = dwt.DWTPooling1D('db2')(inp)
```

```python
inp = Layers.Input((1024, 3))
cA, cD = dwt.DWTPooling1D(pywt.Wavelet('db2'))(inp)
```

## Dependencies
 - tensorflow
 - numpy
 - pywt

## Citations

The `pywt` webpage requests citing them in scientific publications, please do so if you use this repo which relies on their wavelet generating functions.
