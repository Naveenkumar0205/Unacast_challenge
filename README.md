# Project Setup and Execution

This guide will walk you through installing the necessary dependencies and running the pipeline.

## 1. Install Requirements

```bash
pip install -r requirements.txt
```

## 2. Run the pipeline
```bash
python prediction_pipeline/main.py
```

- The `file historical_buffer.csv` is used to create lag features for current day and next day. These are used for model updating and forecasting respectively. After running the prediction pipeline the historical buffer gets updates as the data is forecasted for multiple days. In order to run the prediction pipeline again the `file historical_buffer.csv` has to be re-created from the `model_training.ipynb`

- `daily_forecasting_folder` is considered as stoarge for daily incoming data. For each file `total_visitor_count` for next day is predicted as `visitor_count_total_prediction`. The results are appended to `daily_visitation_summary.csv`

