# Climate Prediction - 2021 CIS&CMU Project

## Problem
Predicting future temperature with machine learning methods.

## Dataset
National Centers for Environmental Information
Global Surface Summary of Day Data (GSOD).

Can be retreived through FTP at <ftp://ftp.ncdc.noaa.gov/pub/data/gsod>.

In this project, we are mainly using station `722860` since it has a long
recording history.

## Items
 - [`gsod.py`](https://github.com/myzhang1029/climate/blob/main/gsod.py):
    Dataset helper and common functions.
 - [`00-FFT.ipynb`](https://github.com/myzhang1029/climate/blob/main/00-FFT.ipynb):
    Real Fourier transform analysis of the recurrent period.
 - [`10-EMD.ipynb`](https://github.com/myzhang1029/climate/blob/main/10-EMD.ipynb):
    Complete empirical mode decomposition of periods and trends.
 - [`11-LSTM.ipynb`](https://github.com/myzhang1029/climate/blob/main/11-LSTM.ipynb):
    Building a long short-term memory model.
 - [`12-MLP.ipynb`](https://github.com/myzhang1029/climate/blob/main/12-MLP.ipynb):
    Building a multilayer perceptron model. (Model by [@stevenli-phoenix](https://github.com/stevenli-phoenix))
 - [`14-Baseline_and_LR.ipynb`](https://github.com/myzhang1029/climate/blob/main/14-Baseline_and_LR.ipynb):
    Building a long short-term memory model.
 - [`20-Ensemble.ipynb`](https://github.com/myzhang1029/climate/blob/main/20-Ensemble.ipynb):
    Combining different models for different modes.

## License
Private until the project is finished. TBD.
