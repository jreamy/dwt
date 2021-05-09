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

You can create a model layer similar to other standard keras layers.
```python
>>> from tensorflow.keras import layers as L
>>> from tensorflow.keras import models as M
>>> import dwt
>>>
>>> inp = L.Input((1024, 3))
>>> cA, cD = dwt.DWTPooling1D('db2')(inp)
>>> x = L.Concatenate(axis=-1)([cA, cD])
>>> x = L.Conv1D(3, 5, activation='relu')(x)
>>> model = M.Model(inp, x)
>>> model.summary()
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 1024, 3)]    0
__________________________________________________________________________________________________
dwt_pooling1d (DWTPooling1D)    [(None, 512, 3), (No 0           input_1[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 512, 6)       0           dwt_pooling1d[0][0]
                                                                 dwt_pooling1d[0][1]
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 508, 3)       93          concatenate[0][0]
==================================================================================================
Total params: 93
Trainable params: 93
Non-trainable params: 0
__________________________________________________________________________________________________

```

## Dependencies
 - tensorflow
 - numpy
 - pywt

## Citations

The `pywt` webpage requests citing them in scientific publications, please do so if you use this repo which relies on their wavelet generating functions.
