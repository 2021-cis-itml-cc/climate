# Data helpers
# Reference: ftp://ftp.ncdc.noaa.gov/pub/data/gsod/readme.txt

"""GSOD Dataset helper."""

from os import PathLike
from pathlib import Path
from typing import Any, List, Tuple

import pandas

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any


def sliding_window(dd: NDArray, bsize: int) -> NDArray:
    """Sliding windows from an 1D array."""
    return np.array([dd[1][n:n+bsize] for n in range(len(dd[1]) - bsize + 1)])


class GsodDataset:
    """GSOD Dataset reader and preprocessor.
    
    `basepath`: Path to gsod. The next level should be year folders.
    """
    _basepath: str
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
        (132, 138)   # FRSHTT Indicators (1 = yes, 0 = no/not
                     #            reported) for the occurrence during the
                     #            day of:
                     #            Fog ('F' - 1st digit).
                     #            Rain or Drizzle ('R' - 2nd digit).
                     #            Snow or Ice Pellets ('S' - 3rd digit).
                     #            Hail ('H' - 4th digit).
                     #            Thunder ('T' - 5th digit).
                     #            Tornado or Funnel Cloud ('T' - 6th
                     #            digit).
    ]
    _NAMES: List[str] = ["STN", "WBAN", "DATE", "TEMP", "TEMPCOUNT", "DEWP",
                         "DEWPCOUNT", "SLP", "SLPCOUNT", "STP", "STPCOUNT",
                         "VISIB", "VISIBCOUNT", "WDSP", "WDSPCOUNT", "MXSPD",
                         "GUST", "MAX", "MAXFLAG", "MIN", "MINFLAG", "PRCP",
                         "PRCPFLAG", "SNDP", "FRSHTT"]

    def __init__(self, basepath: PathLike):
        self._basepath = Path(basepath)
       
    @staticmethod
    def fix_index(dframe: pandas.DataFrame) -> pandas.DataFrame:
        """Fix missing date indices."""
        new_idx = pandas.date_range(min(dframe.index), max(dframe.index))
        return dframe.reindex(new_idx)

    def read(self, *, stn: str, year: str, wban: str = "?????") -> pandas.DataFrame:
        """Read the files as specified and return a combined DataFrame."""
        return pandas.concat((
            pandas.read_fwf(p, index_col=2, header=1, colspecs=self._COLSPEC,
                            parse_dates=[2], names=self._NAMES,
                            compression="infer")
            for p in self._basepath.glob(f"{year}/{stn}-{wban}-{year}.op*"))
        ).sort_values("DATE")
