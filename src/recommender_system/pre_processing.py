import pandas as pd
import numpy as np

def pre_process_data(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the ratings dataframe.

    Parameters
    ----------
    ratings : pd.DataFrame
        The input ratings dataframe.

    Returns
    -------
    pd.DataFrame
        The pre-processed dataframe.
    """
    # Keep only numeric columns
    ratings = ratings.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"])

    # Drop columns with only one unique value
    to_del = [i for i, v in enumerate(ratings.nunique()) if v == 1]
    ratings.drop(ratings.columns[to_del], axis=1, inplace=True)

    # Replace empty values with NaN
    sum_of_null_values = ratings.isnull().sum().sum()
    if sum_of_null_values > 0:
        ratings.replace("", np.nan, regex=False, inplace=True)

    # Drop timestamp column
    ratings.drop(["timestamp"], axis=1, inplace=True)

    return ratings
