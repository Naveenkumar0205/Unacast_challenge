import logging
from typing import Any

import pandas as pd

ALL_VENUE_IDS = [
    "UN000",
    "UN001",
    "UN002",
    "UN003",
    "UN004",
    "CO000",
    "CO001",
    "CO002",
    "CO003",
    "CO004",
    "CI000",
    "CI001",
    "CI002",
    "CI003",
    "CI004",
]

VENUE_MAPPING = {"UN": "university", "CO": "coffee place", "CI": "cinema"}

PARAMETERS = {
    "colsample_bytree": 0.3,
    "learning_rate": 0.05,
    "max_depth": 5,
    "n_estimators": 50,
    "num_leaves": 10,
}


def swap_times(row: pd.Series) -> pd.Series:
    """
    Swap the visit start and end times if the start time is later than the end time.

    Parameters:
    row (pd.Series): A row of a DataFrame containing 'visit_start_time' and 'visit_end_time' columns.

    Returns:
    pd.Series: The same row with corrected start and end times if they were swapped.
    """
    if row["visit_start_time"] > row["visit_end_time"]:
        row["visit_start_time"], row["visit_end_time"] = (
            row["visit_end_time"],
            row["visit_start_time"],
        )
    return row


def map_venue_type(row: pd.Series) -> Any:
    """
    Map the venue_type based on the prefix of the venue_id if the venue_type is missing.

    Parameters:
    row (pd.Series): A row of a DataFrame containing 'venue_type' and 'venue_id' columns.

    Returns:
    str: The mapped venue type if the mapping exists, otherwise NaN.
    """
    if pd.isna(row["venue_type"]):
        prefix = row["venue_id"][:2]
        if prefix in VENUE_MAPPING:
            return VENUE_MAPPING[prefix]
        else:
            logging.warning(f"Mapping not found for venue_id prefix: {prefix}")
    return row["venue_type"]


def convert_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to datetime format.

    Parameters:
    series (pd.Series): A Series containing date or time values as strings or objects.

    Returns:
    pd.Series: The same Series with values converted to pandas datetime objects.
    """
    return pd.to_datetime(series)
