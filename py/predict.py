#!/usr/bin/env python
# coding: utf-8

import gc
import json
import pickle
from typing import Iterable

import numpy as np
import tensorflow as tf

import gsod

# 20-Ensemble.py

# Load modules

# In[2]:

imfs = np.load("runtime/imfs.npy")
lr_last_imf = pickle.load(open("runtime/last_imf_linear.pkl", "rb"))


# LSTM handles the high-frequency periods and MLP handles the rest.

# In[3]:


def get_models():
    """Automatically select a model based on test MAE."""
    mlp_performace = json.load(open("runtime/mlp_perf.json"))["test"]
    lstm_performace = json.load(open("runtime/lstm_perf.json"))["test"]
    # Skip the last IMF (the trend term)
    for i in range(len(imfs.T) - 1):
        name = f"IMF_{i + 1}"
        if mlp_performace[name][1] < lstm_performace[name][1]:
            print(f"Selecting MLP for {name}")
            yield tf.keras.models.load_model(f"runtime/MLP_{name}.h5")
        else:
            print(f"Selecting LSTM for {name}")
            yield tf.keras.models.load_model(f"runtime/LSTM_{name}.h5")

# Prediction helper for the models.

# In[4]:


def predict_temperature(
    prev_imfs,
    models: Iterable[tf.keras.Model]
) -> float:
    """Predict a single temperature from a window.

    Parameters
    ----------
        prev_imfs: (n_imfs, window_width) IMFs.
        models: List of models for each imf.

    Returns
    -------
    float
        Prediction.
    """
    result = 0.0
    for imf, model in zip(prev_imfs, models):
        model_input = imf.reshape(1, -1, 1)
        result += model(model_input)[0, -1, 0]
    return result


def batch_predict_temperature(
    imfs,
    models: Iterable[tf.keras.Model],
    *,
    window_width: int = 7,
    n_predictions: int = -1
):
    """Predict a single temperature from a window.

    Parameters
    ----------
        imfs: (n_imfs, n_predictions + window_width - 1) IMFs for predictions.
              If the length of the second dimension is larger, the extra ones
              at the front will be ignored.
        models: List of models for each imf.
        window_width: width of the window.
        n_predictions: number of predictions to make. Default of -1 for all.

    Returns
    -------
    NDArray[float]
        Predictions.
    """
    # Reset n_predictions for all
    if n_predictions < 0:
        n_predictions = imfs.shape[1] + 1 - window_width
    # Pre-allocate array
    result = np.zeros(n_predictions)
    for imf, model in zip(imfs, models):
        imf_to_use = imf[-(n_predictions + window_width - 1):]
        window = gsod.sliding_window(imf_to_use, window_width)
        # Add each IMF's result to the prediction
        result += model.predict(
            window.reshape(-1, window_width, 1)
            # window_width - shift
        )[:, window_width - 1, 0]
        tf.keras.backend.clear_session()
        gc.collect()
    return result


# In[5]:


with tf.device("/cpu:0"):
    result = batch_predict_temperature(np.asarray(imfs.T)[:-1], get_models())


# ## Linear Regression Handles the Trend

# In[6]:


last_imfs_windows = gsod.sliding_window(imfs[imfs.columns[-1]], 7)
last_imf = lr_last_imf.predict(last_imfs_windows)


# ## Finally Combine the results

# In[7]:


# The last temperature is the future, so skip its calculation
prediction = result[:-1] + last_imf[:-1]
