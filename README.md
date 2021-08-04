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
 - `00-FFT.ipynb`: Real Fourier transform analysis of the recurrent
    period.
 - `10-EMD.ipynb`: Complete empirical mode decomposition of periods
    and trends.
 - `11-LSTM.ipynb`: Building a long short-term memory model.
 - `20-Ensemble.ipynb`: Combining different models for different modes.
