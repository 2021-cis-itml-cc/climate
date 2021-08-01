# cython: language_level=3, boundscheck=False, 
# Cython Extension for Climate Markov Prediction
# Copyright (C) 2021 Zhang Maiyun.

import pickle
from typing import Any, Generator

import numpy as np
import scipy.optimize as so
from numpy.typing import NDArray
from scipy.stats import norm

cimport numpy as np
cimport scipy.optimize as so

cdef class Station:
    _f: str
    _stations: NDArray[np.uint32]
    _temperatures: NDArray[np.float64]
    _dates: NDArray[np.datetime64]

    def __cinit__(self, file: str):
        self._f = file
        self._read()

    cdef _read(self):
        """Read and parse data file."""
        cdef list stations = []
        cdef list temperatures = []
        cdef list dates = []
        cdef int n
        cdef str line

        for n, line in enumerate(open(self._f, mode="rt")):
            if n == 0:
                # Skip header
                continue
            # Use splitlines to avoid killing noeol files
            line = line.splitlines()[0]
            stations.append(np.uint32(line[0:6]))
            temperatures.append(np.float64(line[24:30]))
            dates.append(np.datetime64(
                f"{line[14:18]}-{line[18:20]}-{line[20:22]}"))
        self._stations = np.array(stations)
        self._temperatures = np.array(temperatures)
        self._dates = np.array(dates)

    cpdef np.ndarray[np.float64_t, ndim=1] get_temperatures(self): # -> NDArray[np.floating]:
        """Get the temperature data in this file."""
        return self._temperatures

    cpdef np.ndarray[np.float64_t, ndim=1] get_dates(self): # -> NDArray[np.datetime64]:
        """Get the dates in this file."""
        return self._dates

    cpdef np.ndarray[np.uint32_t, ndim=1] get_date_diffs_jan1(self): # -> NDArray[np.integer]:
        """Get the dates as days from Jan 1st in this file."""
        return np.array([
            date - date.astype("datetime64[Y]")
            for date in self._dates
        ]).astype(np.uint32)


cdef class NDTemperaturePredict:
    """Predict temperatures with customizable n-degree Markov methods."""
    cdef np.ndarray _xs # (n, degree)
    cdef np.ndarray _ys # (n,)
    cdef np.ndarray _absolute_temperatures # (n,)
    cdef np.ndarray _dates # (n,) corresponding to _xs.T[-1]
    cdef np.ndarray _weights # (n, degree), scaled so that they sum to one
    cdef np.uint32_t _len # n
    cdef float _sigma
    cdef int _degree

    def __cinit__(self, int degree = 7):
        self._xs = np.array([]).reshape(0, degree)
        self._ys = np.array([])
        self._dates = np.array([], dtype=np.uint32)
        self._weights = np.asarray([1 / degree] * degree)
        self._len = 0
        # Approximately a month around
        self._sigma = 15.0
        self._degree = degree

    cpdef int get_degree(self) except -1:
        """Getter for self._degree."""
        return self._degree
    
    cpdef put_data(self, dataset: Station):
        """Add data."""
        cdef np.ndarray[np.float64_t, ndim=2] tmp = np.array(list(self.normalized_degree_wise(dataset.get_temperatures())))
        self._len += len(tmp)
        self._xs = np.concatenate((self._xs, tmp.T[:-1].T))
        self._ys = np.concatenate((self._ys, tmp.T[-1]))
        self._dates = np.concatenate((
            self._dates,
            dataset.get_date_diffs_jan1()[7:]
        ))

    cpdef set_stdev(self, float sigma):
        """Set date range standard deviation.
        
        `sigma`: Gaussian stdev.
        """
        self._sigma = sigma
        
    cpdef set_weights(self, weights: NDArray[np.floating]):
        """Set distance weights.
        
        `weights`: Weights.
        """
        self._weights = weights / weights.sum()

    cdef tuple normalize(self, np.ndarray[np.float64_t, ndim=1] iterable, bint label = False):# -> Tuple[NDArray[Any], Any, Any]
        """Normalize a temperature vector to [-1, 1]. If `label` is True, the last
        element is used as a label. However, it is not considered in the
        normalization process, but instead normalized with the same parameters.

        Return the result, scale and offset.
        i.e. Original = Returned * Scale + Offset."""
        cdef np.float64_t offset, scale
        cdef np.ndarray[np.float64_t, ndim=1] masked
        offset = iterable[:-1].mean() if label else iterable.mean()
        # -= is not & = & - here. Do NOT overwrite iterable.
        iterable = iterable - offset
        # Lightweight mask since NDArray slicing is non-cloning
        masked = iterable[:-1] if label else iterable
        scale = np.abs(masked).max()
        return (iterable / scale, scale, offset)

    def _degree_wise(self, iterable: NDArray[Any]) -> Generator[NDArray[Any], None, None]:
        """Group every `self._degree` of an iterable as a generator.

        Also includes the next element as the label."""
        for idx in range(len(iterable) - self._degree):
            yield iterable[idx:(idx + self._degree + 1)]

    def normalized_degree_wise(self, iterable: NDArray[Any]) -> Generator[NDArray[Any], None, None]:
        """Group every `self._degree` of an iterable as a generator.

        Also includes the next element as the label.
        The results are fed into `self._normalize` before returned."""
        for m in self._degree_wise(iterable):
            yield self.normalize(m, True)[0]

    cpdef np.float64_t distance(
        self,
        line1: NDArray[np.floating],
        line2: NDArray[np.floating]
    ) except -1: # -> np.floating
        """Calculate weighted Euclidean distance between two 1D lines."""
        cdef np.float64_t sum_l2_norms = (self._weights * (line1 - line2) ** 2).sum()
        return sum_l2_norms

    cpdef np.float64_t loocv_loss(self): # -> np.floating
        """LOOCV average loss function."""
        cdef np.ndarray[np.float64_t, ndim=1] errors = np.zeros(self._len)
        cdef np.ndarray[np.uint8_t, ndim=1] data_mask = np.zeros(self._len, dtype='B')
        cdef np.ndarray[np.float64_t, ndim=1] loo
        cdef np.float64_t prediction
        cdef Py_ssize_t n
        for n in range(self._len):
            loo = self._xs[n]
            # Mask this item to LOO
            data_mask[n] = 1
            prediction = self._raw_predict(
                loo,
                self._dates[n],
                mask=data_mask
            )
            errors[n] = (self._ys[n] - prediction) ** 2
            # Clear mask
            data_mask[n] = 0
        return errors.mean()
    
    cpdef np.float64_t _optimizer_loss(
        self,
        weights: NDArray[np.floating]
    ):
        """Returning np.floating to match precision."""
        self.set_weights(weights)
        cdef np.float64_t loss = self.loocv_loss()
        print(f"Current loss: {loss}, weights: {weights}", end="\r")
        return loss

    cpdef train(self, maxiter: int = -1):
        """Train the internal dataset. """
        cdef list bounds = [(-1.0,1.0)] * self._degree
        cdef dict options = {"disp": True}
        if maxiter > 0:
             options["maxiter"] = maxiter
        # SLSQP/L-BFGS-B
        so.minimize(
            self._optimizer_loss,
            self._weights,
            method="L-BFGS-B",
            bounds=bounds,
            options=options
        )

    cdef np.float64_t _raw_predict(
        self,
        np.ndarray past_week_temperature,
        np.uint32_t date,
        np.ndarray mask = None
     ) except -999:
        """Predict a temperature change for the following day without normalization.

        `past_week_temperature`: Temperatures for the past seven days. Shape: (self._degree,)
        `date`: Date representation of `past_week_temperature[-1]`.
        `mask`: Mask data for evaluation. Shape: (n, )

        Return: Temperature prediction.
        """
        # The weights from the date difference
        cdef np.ndarray[np.float64_t, ndim=1] date_weight = np.ma.masked_array(
            data=norm(date, self._sigma).pdf(self._dates),
            mask=mask
        )
        # The weights from the distance difference
        cdef np.ndarray[np.float64_t, ndim=1] dist_weight = np.add.reduce(
            self._weights * (past_week_temperature - self._xs) ** 2,
            axis=1
        )
        return np.average(self._ys, weights = date_weight * dist_weight)
    
    cpdef np.float64_t predict(
        self,
        np.ndarray past_week_temperature,
        np.uint32_t date,
        np.ndarray mask = None
    ) except -999:
        """Predict a temperature change for the following day after normalization.

        `past_week_temperature`: Temperatures for the past seven days. Shape: (self._degree,)
        `date`: Date representation of `past_week_temperature[-1]`.
        `mask`: Mask data for evaluation. Shape: (n, )

        Return: Temperature prediction.
        """
        normd, scale, offset = self.normalize(past_week_temperature)
        result = self._raw_predict(normd, date, mask)
        return result * scale + offset
