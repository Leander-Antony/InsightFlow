import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

def detect_problem_type(file_path: str, target_column: str) -> str:
    df = pd.read_csv(file_path)

    # Try converting each column to datetime
    for col in df.columns:
        sample_value = str(df[col].iloc[0])
        if any(char.isdigit() for char in sample_value) and ("-" in sample_value or "/" in sample_value):
            try:
                df[col] = pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
            except:
                continue


    # === Unsupervised FIRST ===
    if target_column.lower() in ["", "none", "unsupervised", "na"]:
        return "Unsupervised Learning"

    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset.")

    target = df[target_column]

    # === Check for time series ===
    date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
    if date_cols:
        if df[date_cols[0]].is_monotonic_increasing:
            return f"Time Series Forecasting on '{target_column}'"

    # === Classification ===
    if not is_numeric_dtype(target):
        return "Classification"

    unique_vals = target.nunique()

    if pd.api.types.is_integer_dtype(target) and unique_vals <= 20:
        return "Classification"

    if is_numeric_dtype(target) and unique_vals > 20:
        return "Regression"

    return "Unable to confidently determine problem type (check target values)"


# # === Example Usage ===
# print(detect_problem_type("test_classification.csv", "habitat"))
# print(detect_problem_type("test_linreg.csv", "belly"))
# print(detect_problem_type("test_simplerregression.csv", "y"))
# print(detect_problem_type("test_timeseries.csv", "humidity"))
# print(detect_problem_type("test_unsupervised.csv", ""))
