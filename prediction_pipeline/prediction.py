import logging
import math

import lightgbm as lgb
import pandas as pd
from utils import PARAMETERS, convert_to_datetime


def model_update(current_data: pd.DataFrame) -> pd.Series:
    """
    Update the model using the current vistor count data.

    Parameters:
    current_data (pd.DataFrame): A DataFrame containing the latest data for model updating.

    Returns:
    pd.Series: A Series representing the forecast of visitor count from updated model.
    """
    # target data for model update
    y_new = current_data["visitor_count_total"]
    # current data - venue vistors info calculated today
    current_data["date"] = convert_to_datetime(current_data["date"])
    # forecasting visitor for venues for next day
    forecast_data = current_data[["date", "venue_id"]]
    forecast_data["date"] = forecast_data["date"] + pd.Timedelta(days=1)

    # data used for training lightgbm model. contains only data for past 7 days to create lag based features
    historical_buffer_data = pd.read_csv("historical_buffer.csv")
    historical_buffer_data["date"] = convert_to_datetime(historical_buffer_data["date"])

    historical_buffer_data = pd.concat(
        [historical_buffer_data, current_data, forecast_data], ignore_index=True
    )

    # create lag features from 1 to 6 days
    for lag in range(1, 7):
        historical_buffer_data[f"Lag_{lag}"] = historical_buffer_data.groupby(
            "venue_id"
        )["visitor_count_total"].shift(lag)

    # extract date based features
    historical_buffer_data["Day_of_Week"] = historical_buffer_data["date"].dt.dayofweek
    historical_buffer_data["day"] = historical_buffer_data["date"].dt.day
    historical_buffer_data["Month"] = historical_buffer_data["date"].dt.month
    historical_buffer_data["Year"] = historical_buffer_data["date"].dt.year
    historical_buffer_data["venue_id"] = historical_buffer_data["venue_id"].astype(
        "category"
    )

    # features for current day and next day created from lags and date values
    training_features = historical_buffer_data[
        historical_buffer_data["date"] == current_data["date"][0]
    ]
    forecast_features = historical_buffer_data[
        historical_buffer_data["date"] == forecast_data["date"][0]
    ]

    # update the historical buffer data with current day visitor counts
    historical_buffer_data = historical_buffer_data[
        (historical_buffer_data["date"] != historical_buffer_data["date"].min())
        & (historical_buffer_data["date"] != historical_buffer_data["date"].max())
    ]
    historical_buffer_data[["date", "venue_id", "visitor_count_total"]].to_csv(
        "historical_buffer.csv", header=True, index=False
    )
    logging.info("Updated historical buffer data")

    logging.info("starting lightgbm model update (online training)")
    new_train_data = lgb.Dataset(
        training_features.drop(columns=["date", "visitor_count_total"]), label=y_new
    )  # Initialize dataset to update the lightgbm model
    incremental_model = lgb.train(
        PARAMETERS,
        new_train_data,
        num_boost_round=50,
        init_model="lightgbm_model/lightgbm_model.txt",
    )  # model update
    incremental_model.save_model("lightgbm_model/lightgbm_model.txt")
    logging.info(
        "successfully updated the model with today's vistor count across venues"
    )

    # forecast visitor count for tomorrow
    lightbgm_predictions = visitor_forecast(forecast_features)

    return lightbgm_predictions


def visitor_forecast(features_df: pd.DataFrame) -> pd.Series:
    """
    Generate visitor forecasts based on the provided feature data.

    Parameters:
    features_df (pd.DataFrame): A DataFrame containing the input features required for forecasting visitor counts.

    Returns:
    pd.Series: A Series containing the predicted visitor counts for each venue id.
    """
    # load lightgbm model for forecasting
    lgb_model = lgb.Booster(model_file="lightgbm_model/lightgbm_model.txt")
    features_df["predictions"] = lgb_model.predict(
        features_df.drop(columns=["date", "visitor_count_total"])
    )  # predictions for next day
    features_df["predictions"] = features_df["predictions"].apply(
        lambda x: math.ceil(x)
    )
    logging.info(
        "Successfully forecasted total number of expected visitors across venues for next date"
    )

    return features_df["predictions"].reset_index(drop=True)
