import logging
import os
from pathlib import Path

import pandas as pd
from data_pre_process import pre_process_venue_data

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    forecast_folder = Path.cwd().joinpath("daily_forecasting_files")

    for file in os.listdir(forecast_folder):
        file_path = os.path.join(forecast_folder, file)
        forecast_result_df = pre_process_venue_data(file_path)

        results_file = Path.cwd().joinpath("daily_visitation_summary.csv")

        if results_file.exists():
            daily_visitation_df = pd.read_csv(results_file)
            daily_visitation_df = pd.concat(
                [daily_visitation_df, forecast_result_df], ignore_index=True
            )
            daily_visitation_df.to_csv(
                "daily_visitation_summary.csv", header=True, index=False
            )
        else:
            forecast_result_df.to_csv(
                "daily_visitation_summary.csv", header=True, index=False
            )
