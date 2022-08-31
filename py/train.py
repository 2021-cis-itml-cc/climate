#!/usr/bin/env python
# coding: utf-8

# 10-EMD.py

# # Empirical Mode Decomposition Analysis

# In[1]:


import gc
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
# Using https://bitbucket.org/luukko/libeemd.git
# and its Python binding https://bitbucket.org/luukko/pyeemd.git
import pyeemd as eemd
import tensorflow as tf
from sklearn.linear_model import LinearRegression

import gsod

# In[2]:


STATION = "722860"
ds = gsod.GsodDataset("runtime/GSOD")
filled = ds.read_continuous(stn=STATION, year="????", fill="ffill")["TEMP"]
indices = filled.index
d = np.asarray(filled)


# In[3]:


imfs_array = eemd.ceemdan(d)


# ## Save
# Save decomposed temperatures for following procedures.

# In[8]:


np.save("runtime/imfs.npy", imfs_array)


# 11-LSTM.py

# # Predicting Individual IMFs with LSTM Networks

# In[3]:


imfs = pd.DataFrame(imfs_array.T,
                    columns=[f"IMF_{n+1}" for n in range(len(imfs_array))])
imfs.index = filled.index
dataframe = pd.merge(filled, imfs, how="left",
                     left_index=True, right_index=True)


# Specification for the LSTM model, training parameters, and helper functions.

# In[4]:


WIDTH = 7
MAX_EPOCHS = 100
N_BATCH = 64
# Features other than the IMF
# "DEWP", , "STP", "VISIB", "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", "SNDP"
OTHER_FEATURES: List[str] = []


def build_compile_model_lstm() -> tf.keras.Model:
    """Specify and compile a model."""
    input_layer = tf.keras.Input(
        # Enable to use stateful model
        # batch_shape=(N_BATCH, WIDTH, len(OTHER_FEATURES) + 1)
        shape=(WIDTH, len(OTHER_FEATURES) + 1)
    )
    lstm = tf.keras.layers.Dense(
        N_BATCH * WIDTH * (len(OTHER_FEATURES) + 1), activation="tanh"
    )(input_layer)
    # Shape [batch, time, features] => [batch, time, lstm_units]
    lstm = tf.keras.layers.LSTM(
        N_BATCH, return_sequences=True,
        # stateful=True,
        dropout=0., recurrent_dropout=0.3
    )(lstm)
    lstm = tf.keras.layers.LSTM(
        N_BATCH, return_sequences=True,
        # stateful=True,
        dropout=0., recurrent_dropout=0.2
    )(lstm)
    # Shape => [batch, time, features]
    lstm = tf.keras.layers.Dense(units=1)(lstm)
    model = tf.keras.Model(inputs=input_layer, outputs=lstm)
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    return model


def fit(model, window, patience=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"
    )
    history = model.fit(
        window.train, epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping],
        verbose=0, shuffle=False
    )
    return history


# In[5]:


val_performance = {}
performance = {}
save_path_template = "runtime/LSTM_{imf_name}.h5"
no_reload = False


# Train each IMFs with its own models. In my case, CPU is faster.

# In[6]:


with tf.device("/cpu:0"):
    # Skip the last (trend) IMF
    for imf_name in (f"IMF_{n}" for n in range(1, len(imfs.T))):
        save_path = None
        if save_path_template is not None:
            save_path = Path(save_path_template.format(imf_name=imf_name))
        print(f"Training {imf_name}")
        wide_window = gsod.WindowGenerator(
            df=dataframe[[imf_name] + OTHER_FEATURES],
            input_width=WIDTH, label_width=WIDTH, shift=1,
            batch_size=N_BATCH, label_columns=[imf_name]
        )
        if save_path is not None and save_path.exists() and not no_reload:
            lstm_model = tf.keras.models.load_model(save_path)
        else:
            lstm_model = build_compile_model_lstm()
            fit(lstm_model, wide_window)
        print(f"Validating {imf_name}")
        val_performance[f"{imf_name}"] = lstm_model.evaluate(wide_window.val)
        print(f"Testing {imf_name}")
        performance[f"{imf_name}"] = lstm_model.evaluate(wide_window.test,
                                                         verbose=0)
        wide_window.plot(model=lstm_model, max_subplots=1, plot_col=imf_name,
                         dataset="test", network_name="LSTM", station_name=STATION)
        if save_path is not None:
            lstm_model.save(save_path)
        tf.keras.backend.clear_session()
        gc.collect()


# Save the metrics for comparison

# In[8]:


json.dump({"val": val_performance, "test": performance},
          open("runtime/lstm_perf.json", "w"))


# 12-MLP.py

# In[4]:

def build_compile_model_mlp() -> tf.keras.Model:
    """Specify and compile a model."""
    input_layer = tf.keras.Input(
        # Enable to use stateful model
        # batch_shape=(N_BATCH, WIDTH, len(OTHER_FEATURES) + 1)
        shape=(WIDTH, len(OTHER_FEATURES) + 1)
    )
    mlp = tf.keras.layers.Dense(128, activation="relu")(input_layer)
    mlp = tf.keras.layers.Dense(128, activation="relu")(mlp)
    mlp = tf.keras.layers.Dense(128, activation="relu")(mlp)
    mlp = tf.keras.layers.Dropout(0.2)(mlp)
    mlp = tf.keras.layers.Dense(64, activation="relu")(mlp)
    mlp = tf.keras.layers.Dense(1, activation="relu")(mlp)
    model = tf.keras.Model(inputs=input_layer, outputs=mlp)
    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )
    return model


# In[5]:


val_performance = {}
performance = {}
save_path_template = "runtime/MLP_{imf_name}.h5"
no_reload = False


# Train each IMFs with its own models. In my case, CPU is faster.

# In[6]:


with tf.device("/cpu:0"):
    # Skip the last (trend) IMF
    for imf_name in (f"IMF_{n}" for n in range(1, len(imfs.T))):
        save_path = None
        if save_path_template is not None:
            save_path = Path(save_path_template.format(imf_name=imf_name))
        print(f"Training {imf_name}")
        wide_window = gsod.WindowGenerator(
            df=dataframe[[imf_name] + OTHER_FEATURES],
            input_width=WIDTH, label_width=WIDTH, shift=1,
            batch_size=N_BATCH, label_columns=[imf_name]
        )
        if save_path is not None and save_path.exists() and not no_reload:
            lstm_model = tf.keras.models.load_model(save_path)
        else:
            lstm_model = build_compile_model_mlp()
            fit(lstm_model, wide_window)
        print(f"Validating {imf_name}")
        val_performance[f"{imf_name}"] = lstm_model.evaluate(wide_window.val)
        print(f"Testing {imf_name}")
        performance[f"{imf_name}"] = lstm_model.evaluate(wide_window.test,
                                                         verbose=0)
        wide_window.plot(model=lstm_model, max_subplots=1, plot_col=imf_name,
                         dataset="test", network_name="MLP", station_name=STATION)
        if save_path is not None:
            lstm_model.save(save_path)
        tf.keras.backend.clear_session()

# Save the metrics for comparison

# In[8]:


json.dump({"val": val_performance, "test": performance},
          open("runtime/mlp_perf.json", "w"))


# Linear regression does not seem to perform well.
# However, it is already a better predictor than the baseline.
# Furthermore, it is also a good predictor for the last IMF.

# ## Linearly Predict the Last IMF

# In[6]:


lr_last_imf = LinearRegression()
windows = gsod.sliding_window(imfs_array[-1], 7)[:-1]
length = len(windows)
Y = imfs_array[-1][7:]
Xtrain = windows[:int(length * 0.7)]
Ytrain = Y[:int(length * 0.7)]
lr_last_imf.fit(Xtrain, Ytrain)


# In[7]:


pickle.dump(lr_last_imf, open("runtime/last_imf_linear.pkl", "wb"))
