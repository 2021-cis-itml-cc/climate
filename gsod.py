
"""GSOD Dataset helper."""

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any


def enable_svg_graphing():
    """Enable matplotlib inline SVG graphs."""
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats("svg")


def suppress_tf_log():
    """Suppress extra TensorFlow logs."""
    import logging
    import os

    import tensorflow as tf
    from absl import logging as alogger
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel(logging.FATAL)
    alogger.set_verbosity(alogger.FATAL)


def sliding_window(seq: NDArray, width: int) -> NDArray:
    """Sliding windows from an 1D array.

    Parameters
    ----------
    seq : NDArray
        The sequence to separate
    width : int
        Width of the window

    Returns
    -------
    NDArray
        Separated windows with shape (len(seq) - width + 1, width)

    Examples
    --------
    >>> import numpy as np
    >>> from gsod import sliding_window
    >>> sequence = np.arange(10)
    >>> sliding_window(sequence, 5)
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6],
           [3, 4, 5, 6, 7],
           [4, 5, 6, 7, 8],
           [5, 6, 7, 8, 9]])

    See Also
    --------
    tensorflow.keras.preprocessing.timeseries_dataset_from_array :
        Creates a dataset of sliding windows over a timeseries
        provided as array.
    """
    return np.array([dd[n:n+bsize] for n in range(len(dd) - bsize + 1)])


class GsodDataset:
    """GSOD Dataset reader and preprocessor.

    Parameters
    ----------
    basepath : PathLike
        Path to GSOD. The next level should be year folders.

    References
    ----------
    .. [1] `GSOD Dataset Description
        <ftp://ftp.ncdc.noaa.gov/pub/data/gsod/readme.txt>`_
    """

    _basepath: Path
    # Specification of the dataset files.
    _COLSPEC: List[Tuple[int, int]] = [
        (0,   6),    # STN--- WMO/DATSAV3 Station number
        (7,   12),   # WBAN   Weather Bureau Air Force Navy number
        (14,  22),   # Y4M2D2 Year Month and Day
        (24,  30),   # TEMP   Mean temperature in degrees Fahrenheit
        (31,  33),   # Count  Number of observations
        (35,  41),   # DEWP   Mean dew point in degrees Fahrenheit
        (42,  44),   # Count  Number of observations
        (46,  52),   # SLP    Mean sea level pressure in millibars
        (53,  55),   # Count  Number of observations
        (57,  63),   # STP    Mean station pressure in millibars
        (64,  66),   # Count  Number of observations
        (68,  73),   # VISIB  Mean visibility in miles
        (74,  76),   # Count  Number of observations
        (78,  83),   # WDSP   Mean wind speed in knots
        (84,  86),   # Count  Number of observations
        (88,  93),   # MXSPD  Maximum sustained wind speed reported in knots
        (95,  100),  # GUST   Maximum wind gust reported in knots
        (102, 108),  # MAX    Maximum temperature reported during the day
                     #        in Fahrenheit
        (108, 109),  # Flag   * indicates max temp was derived from
                     #        the hourly data
        (110, 116),  # MIN    Minimum temperature reported during the day
                     #        in Fahrenheit
        (116, 117),  # Flag   * indicates min temp was derived from
                     #        the hourly data
        (118, 123),  # PRCP   Total precipitation (rain and/or melted snow)
                     #        reported during the day in inches and hundredths
        (123, 124),  # Flag   A = 1 report of 6-hour precipitation
                     #            amount.
                     #        B = Summation of 2 reports of 6-hour
                     #            precipitation amount.
                     #        C = Summation of 3 reports of 6-hour
                     #            precipitation amount.
                     #        D = Summation of 4 reports of 6-hour
                     #            precipitation amount.
                     #        E = 1 report of 12-hour precipitation
                     #            amount.
                     #        F = Summation of 2 reports of 12-hour
                     #            precipitation amount.
                     #        G = 1 report of 24-hour precipitation
                     #            amount.
                     #        H = Station reported '0' as the amount
                     #            for the day (eg, from 6-hour reports),
                     #            but also reported at least one
                     #            occurrence of precipitation in hourly
                     #            observations--this could indicate a
                     #            trace occurred, but should be considered
                     #            as incomplete data for the day.
                     #        I = Station did not report any precip data
                     #            for the day and did not report any
                     #            occurrences of precipitation in its hourly
                     #            observations--it's still possible that
                     #            precip occurred but was not reported.
        (125, 130),  # SNDP   Snow depth in inches
                     # FRSHTT Indicators (1 = yes, 0 = no/not
                     #            reported) for the occurrence during the
                     #            day of:
        (132, 133),  # Fog ('F' - 1st digit).
        (133, 134),  # Rain or Drizzle ('R' - 2nd digit).
        (134, 135),  # Snow or Ice Pellets ('S' - 3rd digit).
        (135, 136),  # Hail ('H' - 4th digit).
        (136, 137),  # Thunder ('T' - 5th digit).
        (137, 138)  # Tornado or Funnel Cloud ('T' - 6th
        #            digit).
    ]
    # Names to use for the pandas DataFrame columns.
    _NAMES: List[str] = ["STN", "WBAN", "DATE", "TEMP", "TEMP_COUNT", "DEWP",
                         "DEWP_COUNT", "SLP", "SLP_COUNT", "STP", "STP_COUNT",
                         "VISIB", "VISIB_COUNT", "WDSP", "WDSP_COUNT", "MXSPD",
                         "GUST", "MAX", "MAX_FLAG", "MIN", "MIN_FLAG", "PRCP",
                         "PRCP_FLAG", "SNDP", "FOG", "RAIN", "SNOW", "HAIL",
                         "THUNDER", "TORNADO"]
    # Datatypes to use for the columns.
    _DTYPES: Dict[str, str] = {
        "STN": "uint32",
        "WBAN": "int64",
        "TEMP": "float64",
        "TEMP_COUNT": "uint8",
        "DEWP": "float64",
        "DEWP_COUNT": "uint8",
        "SLP": "float64",
        "SLP_COUNT": "uint8",
        "STP": "float64",
        "STP_COUNT": "uint8",
        "VISIB": "float64",
        "VISIB_COUNT": "uint8",
        "WDSP": "float64",
        "WDSP_COUNT": "uint8",
        "MXSPD": "float64",
        "GUST": "float64",
        "MAX": "float64",
        "MAX_FLAG": "U1",
        "MIN": "float64",
        "MIN_FLAG": "U1",
        "PRCP": "float64",
        "PRCP_FLAG": "U1",
        "SNDP": "float64",
        "FOG": "bool",
        "RAIN": "bool",
        "SNOW": "bool",
        "HAIL": "bool",
        "THUNDER": "bool",
        "TORNADO": "bool"
    }

    def __init__(self, basepath: PathLike):
        self._basepath = Path(basepath)

    @staticmethod
    def fix_index(dframe: pandas.DataFrame) -> pandas.DataFrame:
        """Fix missing date indices.

        Parameters
        ----------
        dframe : DataFrame
            DataFrame with a DateIndex to be fixed.

        Returns
        -------
        DataFrame
            The DataFrame after being fixed.
        """
        new_idx = pandas.date_range(min(dframe.index), max(dframe.index))
        return dframe.reindex(new_idx)

    def read_at(self, path: PathLike) -> pandas.DataFrame:
        """Read the file at `path`.

        Parameters
        ----------
        path : PathLike
            Path to the dataset file.

        Returns
        -------
        DataFrame
            The read table as-is.
        """
        return pandas.read_fwf(path, index_col=2, header=1, dtype=self._DTYPES,
                               colspecs=self._COLSPEC, parse_dates=[2],
                               names=self._NAMES, compression="infer")

    def read(self, *, stn: str, year: str = "????",
             wban: str = "?????") -> pandas.DataFrame:
        """Read the files as specified.

        Parameters
        ----------
        stn : str
            WMO/DATSAV3 Station number as a 6-char string.
        year : str
            Year as a 4-char string.
        wban : str, optional
            Weather Bureau Air Force Navy number. Default: all.
            If specified, it must match the given `stn`.

        Returns
        -------
        DataFrame
            Combined DataFrame from all matched files, sorted by date.
        """
        return pandas.concat((
            self.read_at(p)
            for p in self._basepath.glob(f"{year}/{stn}-{wban}-{year}.op*"))
        ).sort_values("DATE")

    def read_continuous(self, *, stn: str, year: str = "????",
                        wban: str = "?????",
                        interpolate: bool = False) -> pandas.DataFrame:
        """Read the files as specified and make the index continuous.

        Parameters
        ----------
        stn : str
            WMO/DATSAV3 Station number as a 6-char string.
        year : str
            Year as a 4-char string.
        wban : str, optional
            Weather Bureau Air Force Navy number. Default: all.
            If specified, it must match the given `stn`.
        interpolate : bool
            Whether to linearly interpolate missing datapoints.

        Returns
        -------
        DataFrame
            Combined DataFrame from all matched files, sorted by date.
        """
        fixed = self.fix_index(self.read(stn=stn, year=year, wban=wban))
        return fixed.interpolate() if interpolate else fixed
