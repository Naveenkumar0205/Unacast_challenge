import logging

import numpy as np
import pandas as pd
from prediction import model_update
from utils import (ALL_VENUE_IDS, convert_to_datetime, map_venue_type,
                   swap_times)


def pre_process_venue_data(file: str) -> pd.DataFrame:
    """
    Pre-process venue visit data from the given file.

    Parameters:
    file (str): Path to the input file containing venue data.

    Returns:
    pd.DataFrame: A pre-processed DataFrame with transformed venue data and forecasted visitors for next day.
    """
    venue_data = pd.read_csv(file)
    logging.info(f"Pre-processing file {file}")
    # Drop rows with missing venue_id and visitor_id (error handling)
    venue_data = venue_data.dropna(subset=['venue_id', 'visitor_id'])
    # Drop duplicates
    venue_data = venue_data.drop_duplicates(keep="first").reset_index(drop=True)

    # convert 'visit_start_time', 'visit_end_time' to datetime format
    venue_data["visit_start_time"] = convert_to_datetime(venue_data["visit_start_time"])
    venue_data["visit_end_time"] = convert_to_datetime(venue_data["visit_end_time"])

    # Extract date form 'visit_start_time'
    venue_data["date"] = venue_data["visit_start_time"].dt.date

    # date from file name
    file_date = pd.to_datetime(
        file.split("\\")[-1].split(".")[0], format="%Y%m%d"
    ).date()

    # Check if file contains data from differnt dates
    if venue_data["date"].nunique() == 1 and venue_data["date"][0] == file_date:
        # fill missing values in 'visit_end_time' column (median time spent)
        venue_data["visit_end_time"] = venue_data["visit_end_time"].fillna(
            venue_data["visit_start_time"] + pd.Timedelta(minutes=81)
        )

        # if visit_start_time > visit_end_time swap the variables
        venue_data = venue_data.apply(swap_times, axis=1)

        # replace 'unknown' venue_type with NaN values
        venue_data["venue_type"] = venue_data["venue_type"].replace("unknown", np.nan)

        # map the venue_type based on venue_id
        venue_data["venue_type"] = venue_data.apply(map_venue_type, axis=1)
    else:
        logging.warning("file contains data from multiple dates or other date")

    venue_agg_df = (
        venue_data.groupby("venue_id")
        .agg({"date": "first", "visitor_id": ["nunique", "count"]})
        .reset_index()
    )
    venue_agg_df.columns = [
        "venue_id",
        "date",
        "visitor_count_unique",
        "visitor_count_total",
    ]
    venue_agg_df = venue_agg_df[
        ["date", "venue_id", "visitor_count_unique", "visitor_count_total"]
    ]  # rearrange columns

    # Insert 0 as the visitor count for a specific venue if no data is available for that venue.
    missing_venues = set(ALL_VENUE_IDS) - set(venue_agg_df["venue_id"])
    if missing_venues:
        logging.warning(
            f"Missing venue_ids: {missing_venues}, insert 0 as visitor count"
        )

        missing_venues_df = pd.DataFrame(
            {
                "venue_id": list(missing_venues),
                "date": venue_data["date"].iloc[0],  # Use the first date in the dataset
                "visitor_count_unique": 0,
                "visitor_count_total": 0,
            }
        )

        venue_agg_df = pd.concat([venue_agg_df, missing_venues_df], ignore_index=True)

    logging.info(f"pre-processing completed for file {file}")
    # pass the aggregated data for model updating (online training) and forecasting
    next_day_forecast = model_update(
        venue_agg_df[["date", "venue_id", "visitor_count_total"]]
    )
    venue_agg_df["visitor_count_total_prediction"] = next_day_forecast

    return venue_agg_df
