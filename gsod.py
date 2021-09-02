
"""GSOD Dataset helper."""

import fnmatch
import logging
import os
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
from absl import logging as alogger
from matplotlib_inline.backend_inline import set_matplotlib_formats
from tensorflow.python.data.ops.dataset_ops import MapDataset

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any


def enable_svg_graphing():
    """Enable matplotlib inline SVG graphs."""
    set_matplotlib_formats("svg")


def suppress_tf_log():
    """Suppress extra TensorFlow logs."""
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
    return np.array([seq[n:n + width] for n in range(len(seq) - width + 1)])


class GsodDatasetBase:
    """GSOD Dataset reader and preprocessor.

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
    _NAMES: List[str] = ["STN", "WBAN", "DATE", "TEMP", "COUNT_TEMP", "DEWP",
                         "COUNT_DEWP", "SLP", "COUNT_SLP", "STP", "COUNT_STP",
                         "VISIB", "COUNT_VISIB", "WDSP", "COUNT_WDSP", "MXSPD",
                         "GUST", "MAX", "FLAG_MAX", "MIN", "FLAG_MIN", "PRCP",
                         "FLAG_PRCP", "SNDP", "FOG", "RAIN_DRIZZLE",
                         "SNOW_ICE_PELLETS", "HAIL", "THUNDER",
                         "TORNADO_FUNNEL_CLOUD"]
    # Data types to use for the columns.
    _DTYPES: Dict[str, str] = {
        "STN": "uint32",
        "WBAN": "int64",
        "DATE": "datetime64[D]",
        "TEMP": "float64",
        "COUNT_TEMP": "uint8",
        "DEWP": "float64",
        "COUNT_DEWP": "uint8",
        "SLP": "float64",
        "COUNT_SLP": "uint8",
        "STP": "float64",
        "COUNT_STP": "uint8",
        "VISIB": "float64",
        "COUNT_VISIB": "uint8",
        "WDSP": "float64",
        "COUNT_WDSP": "uint8",
        "MXSPD": "float64",
        "GUST": "float64",
        "MAX": "float64",
        "FLAG_MAX": "U1",
        "MIN": "float64",
        "FLAG_MIN": "U1",
        "PRCP": "float64",
        "FLAG_PRCP": "U1",
        "SNDP": "float64",
        "FOG": "bool",
        "RAIN_DRIZZLE": "bool",
        "SNOW_ICE_PELLETS": "bool",
        "HAIL": "bool",
        "THUNDER": "bool",
        "TORNADO_FUNNEL_CLOUD": "bool"
    }

    @staticmethod
    def fix_index(dframe: pandas.DataFrame, **kwargs) -> pandas.DataFrame:
        """Fix missing date indices.

        Parameters
        ----------
        dframe : DataFrame
            DataFrame with a DateIndex to be fixed.
        **kwargs : dict
            Following arguments will be passed to DataFrame.reindex.

        Returns
        -------
        DataFrame
            The DataFrame after being fixed.

        See Also
        --------
        pandas.DataFrame.reindex :
            Conform Series/DataFrame to new index with optional filling logic.
        """
        new_idx = pandas.date_range(min(dframe.index), max(dframe.index))
        return dframe.reindex(new_idx, **kwargs)

    def read(self, *, stn: str, year: str = "????",
             wban: str = "?????") -> pandas.DataFrame:
        """Proxy function for child classes to implement."""
        raise NotImplementedError(
            "Call a child implementation instead of this class")

    def read_continuous(self, *, stn: str, year: str = "????",
                        wban: str = "?????", interpolate: bool = False,
                        fill: Optional[str] = None) -> pandas.DataFrame:
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
        fill : str, optional
            Method of filling missing datapoints: "ffill", "bfill", or None.
            If None is specified, some fields will be converted to float.

        Returns
        -------
        DataFrame
            Combined DataFrame from all matched files, sorted by date.
        """
        fixed = self.fix_index(
            self.read(stn=stn, year=year, wban=wban), method=fill)
        return fixed.interpolate() if interpolate else fixed


class GsodDiskDataset(GsodDatasetBase):
    """GSOD Dataset on disk.

    Parameters
    ----------
    basepath : PathLike
        Path to GSOD. The next level should be year folders.
    """

    def __init__(self, basepath: PathLike):
        self._basepath = Path(basepath)

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
        dataframe = pandas.read_fwf(path, index_col=2, header=0,
                                    dtype=self._DTYPES,
                                    colspecs=self._COLSPEC, parse_dates=[2],
                                    names=self._NAMES, compression="infer")
        # read_fwf does not play well with empty U1 fields
        return dataframe.fillna({"FLAG_MAX": " ", "FLAG_MIN": " "})

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
        return pandas.concat(
            self.read_at(p)
            for p in self._basepath.glob(f"{year}/{stn}-{wban}-{year}.op*")
        ).sort_index()


class GsodBigQueryDataset(GsodDatasetBase):
    """GSOD Dataset queries though Google BigQuery.

    Please set appropriate authorization before invoking.

    Warnings
    --------
    Some fields in the BigQuery GSOD dataset are corrupted!
    Namely, some of the STP values are reduced by 1000.0 if the value
    exceeds 1000.0.
    And some of the FRSHTT flags are different from the downloaded dataset.
    Proceed with care for those fields.

    See Also
    --------
    google.auth.default :
        Authentication to Google APIs
        <https://googleapis.dev/python/google-api-core/latest/auth.html>

    References
    ----------
    .. [1] `Kaggle NOAA GSOD Metadata
        <https://www.kaggle.com/noaa/gsod/metadata>`_
    .. [2] `Google BigQuery Reference
        <https://googleapis.dev/python/bigquery/latest/reference.html>`_
    """

    _QUERY = """SELECT *
    FROM `bigquery-public-data.noaa_gsod.{table_id}`
    WHERE stn LIKE '{stn}' AND wban LIKE '{wban}'"""

    def __init__(self):
        # Defer this import unless required
        # pylint: disable=import-outside-toplevel
        from google.cloud import bigquery
        client = bigquery.Client()
        self._client = client
        gsod_data_ref = client.dataset(
            "noaa_gsod", project="bigquery-public-data")
        gsod_dataset = client.get_dataset(gsod_data_ref)
        self._table_ids = [
            x.table_id for x in client.list_tables(gsod_dataset)]

    def _transform_dataframe(
        self,
        dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Transform a NOAA GSOD BigQuery DataFrame to our format.

        Parameters
        ----------
        dataframe : DataFrame
            DataFrame to operate on.

        Returns
        -------
        DataFrame
            Transformed DataFrame.
        """
        # This line duplicates dataframe
        dataframe = dataframe.drop(columns=["year", "mo", "da"])
        # So it is not overwriten here
        dataframe.columns = dataframe.columns.str.upper()
        # Convert those boolean fields to int
        # - otherwise they'll all be True
        for field in ("FOG", "RAIN_DRIZZLE", "SNOW_ICE_PELLETS", "HAIL",
                      "THUNDER", "TORNADO_FUNNEL_CLOUD"):
            dataframe[field] = dataframe[field].astype("uint8")
        rep = {"None": " "}
        return (dataframe
                # This is a bug in BigQuery NOAA GSOD
                .rename(columns={"MXPSD": "MXSPD"})
                # Convert objects to numeric - makes another copy
                .astype(self._DTYPES)
                # Make this consistent
                .replace({"FLAG_MAX": rep, "FLAG_MIN": rep})
                # Index and sort by date
                .set_index("DATE")
                .sort_index()
                )

    @staticmethod
    def _translate_sql_like(unix_glob: str) -> str:
        """Translate Unix glob to SQL LIKE patterns.

        This method only performs this replacement:
            '?' -> '_', '*' -> '%'

        Parameters
        ----------
        unix_glob : str
            Unix glob pattern.

        Returns
        -------
        str
            SQL LIKE pattern.
        """
        trans_table = str.maketrans({'?': '_', '*': '%'})
        return unix_glob.translate(trans_table)

    def read(self, *, stn: str, year: str = "????",
             wban: str = "?????") -> pandas.DataFrame:
        """Query data as specified.

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
        # Glob the table ids
        matched_table_ids = fnmatch.filter(self._table_ids, "gsod" + year)
        # Convert glob to SQL LIKE
        stn = self._translate_sql_like(stn)
        wban = self._translate_sql_like(wban)
        # Run queries
        data = pandas.concat(
            self._client.query(self._QUERY.format(
                table_id=table_id, stn=stn, wban=wban)).result().to_dataframe()
            for table_id in matched_table_ids
        )
        # Transform the result
        return self._transform_dataframe(data)


class GsodDataset(GsodDatasetBase):
    """Choose a GSOD Dataset automatically.

    Parameters
    ----------
    basepath : PathLike, optional
        Path to GSOD. The next level should be year folders.
    """

    def __init__(self, basepath: PathLike = None):
        if basepath:
            self.inner: GsodDatasetBase = GsodDiskDataset(basepath=basepath)
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            try:
                self.inner = GsodBigQueryDataset()
            except ImportError as imp_error:
                raise ValueError(
                    "Base path nor Google BigQuery available") from imp_error
        else:
            raise ValueError("Base path nor Google authentication available")

    def read(self, *args, **kwargs):
        """Read a GSOD Dataset.

        See Also
        --------
        GsodBigQueryDataset.read :
            read() method of the BigQuery implementor
        GsodDiskDataset.read :
            read() method of the Disk implementor
        """
        return self.inner.read(*args, **kwargs)


# pylint: disable=too-many-instance-attributes
class WindowGenerator:
    """Build a window from the data for training.

       Sliding Window Generator.
       Mostly from [TensorFlow Time Series Example]
       (https://www.tensorflow.org/tutorials/structured_data/time_series),
       but also includes my own comments and modifications.

                       | - - total size - - |
        input_indices: 0 1 2 3 4 5
        output_indices:        4 5 6 7 8 9 10
                       |  input  | - shft - |
                               | - output - |

    Parameters
    ----------
        df : DataFrame
            DataFrame containing the dataset.
        input_width : int
            Width of the feature input.
        label_width : int
            Width of the outputs.
        shift : int
            Shift (of the end) between the input window and the output window.
        batch_size : int
            Size of training batches.
        label_columns : list[str]
            List of columns to be used as the label.

    References
    ----------
    .. [1] `TensorFlow Time Series Example
        <https://www.tensorflow.org/tutorials/structured_data/time_series>`_
    """

    # pylint: disable=too-many-arguments
    def __init__(self, df: pandas.DataFrame, input_width: int,
                 label_width: int, shift: int, batch_size: int,
                 label_columns=None):
        # Split the dataset and store it
        length = len(df)
        self.train_df = df[:int(length*0.7)]               # 70%
        self.val_df = df[int(length*0.7):int(length*0.9)]  # 20%
        self.test_df = df[int(length*0.9):]                # 10%

        # Work out the label column indices.
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.batch_size = batch_size

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        """Print Window information."""
        return "\n".join([
            f"Total window size: {self.total_window_size}",
            f"Input indices: {self.input_indices}",
            f"Label indices: {self.label_indices}",
            f"Label column name(s): {self.label_columns}"])

    def make_dataset(self, data: pandas.DataFrame) -> MapDataset:
        """Generate windowed dataset for training from continuous dataset.

        Parameters
        ---------
        data : DataFrame
            DataFrame containing the continuous dataset.

        Returns
        -------
        MapDataset
            Pair of (input, label)
            where the shape of input is (n, input_width, n_columns)
            and the shape of label is (n, label_width, n_columns)
        """
        data = np.array(data, dtype=np.float32)
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,)
        return dataset.map(self.split_window)

    @property
    def train(self):
        """Make training dataset."""
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """Make validation dataset."""
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """Make testing dataset."""
        return self.make_dataset(self.test_df)

    @lru_cache
    def get_example(self,
                    dataset: str = "train") -> Tuple[tf.Tensor, tf.Tensor]:
        """Get and cache an example batch of `inputs, labels` for plotting.

        Parameters
        ----------
            dataset : str
                One of "train", "val", or "test".

        Returns
        -------
        tuple[Tensor, Tensor]
            Pair of (input, label), respectively of shape
            (batch_size, input_size, n_feature) and
            (batch_size, label_size, n_feature).
        """
        return next(iter(getattr(self, dataset)))

    def split_window(self, features: tf.Tensor) -> Tuple:
        """Magic. Something that I don't understand. Comment & typing TODO."""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(
        self,
        *,
        model: Optional[tf.keras.Model] = None,
        plot_col: Optional[str] = None,
        max_subplots: int = 3,
        dataset: str = "train",
        network_name: Optional[str] = None,
        station_name: Optional[str] = None
    ):
        """Plot the specified dataset and its training results.

        Parameters
        ----------
            model : Model, optional
                Trained model.
            plot_col : str, optional
                Index of the feature column to plot.
            max_subplots : int
                Maximum number of subplots.
            dataset : str
                Name of the set.
            network_name : str, optional
                Name of the network.
            station_name : str, optional
                Name of the station.
        """
        # Generate examples from the specified dataset
        inputs, labels = self.get_example(dataset)
        # Select the first feature if not specified
        plot_col = plot_col if plot_col is not None else list(
            self.column_indices.keys())[0]
        plot_col_index = self.column_indices[plot_col]
        # len(inputs) is batch size
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f"{plot_col}")
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label="Inputs", marker=".", zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors="k", label="Labels", c="#2ca02c", s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker="X", edgecolors="k", label="Predictions",
                            c="#ff7f0e", s=64)
            if n == 0:
                plt.legend()
        plt.suptitle((f"{network_name} Network " if network_name else "")
                     + f"Samples for {plot_col} {dataset.capitalize()} Set"
                     + (f" - Station {station_name}" if station_name else ""))
        plt.xlabel("Days")
