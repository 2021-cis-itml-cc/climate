# Climate Prediction - 2021 CIS&CMU Project

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/2021-cis-itml-cc/climate/HEAD)

## Problem
Predicting future temperature with machine learning methods.

## Dataset
National Centers for Environmental Information
Global Surface Summary of Day Data (GSOD).
See [GSOD README](https://github.com/2021-cis-itml-cc/climate/blob/main/README_GSOD.txt)
for information on how to obtain it and/or its terms and restrictions.

In this project, we are mainly using station `722860` since it has a long
recording history. To run the notebooks, place the dataset at `runtime/GSOD`,
so that `runtime/GSOD/2021` contains `.op` or `.op.gz` data files.
For example, `runtime/GSOD/2021/583620-99999-2021.op.gz` should exist.

## Items
Use those Notebooks by this order to start.
 - [`gsod.py`](https://github.com/2021-cis-itml-cc/climate/blob/main/gsod.py):
    Dataset helper and common functions.
 - [`00-Search_Stations.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/00-Search_Stations.ipynb):
    Some helpers to find good stations (optional).
 - [`01-FFT.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/01-FFT.ipynb):
    Real Fourier transform analysis of the recurrent period (optional).
 - [`10-EMD.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/10-EMD.ipynb):
    Complete empirical mode decomposition of periods and trends.
 - [`11-LSTM.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/11-LSTM.ipynb):
    Building a long short-term memory model.
 - [`12-MLP.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/12-MLP.ipynb):
    Building a multilayer perceptron model. (Model by [@stevenli-phoenix](https://github.com/stevenli-phoenix))
 - [`13-LSTM_Only.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/13-LSTM_Only.ipynb):
    A comparative LSTM-only approach (optional).
 - [`14-Baseline_and_LR.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/14-Baseline_and_LR.ipynb):
    Building a long short-term memory model.
 - [`15-ARIMA_and_SVM.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/15-ARIMA_and_SVM.ipynb):
    Comparative ARIMA and SVM models (optional).
 - [`20-Ensemble.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/20-Ensemble.ipynb):
    Combining different models for different mode functions.

## Historical Models
Early experiments. Kept here for reference.
 - [`ClimateMarkov.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/ClimateMarkov.ipynb):
    First-degree Markov-like model.
 - [`ClimateMarkov2.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/ClimateMarkov2.ipynb)
    Any-degree Markov-like model.
 - [`ClimateDiscreteMarkov.ipynb`](https://github.com/2021-cis-itml-cc/climate/blob/main/ClimateDiscreteMarkov.ipynb)
    Any-degree discrete Markov-like model.

## License
Private until the project is finished. TBD.
