# main.py

import pandas as pd
from prob_type import detect_problem_type
from preprocess import (
    preprocess_regression,
    preprocess_classification,
    preprocess_time_series,
    preprocess_unsupervised,
)
from feature_selection import select_features_model_based
from sklearn.preprocessing import StandardScaler
from eda_utils import full_eda
from model_training import train_and_evaluate, train_and_evaluate_unsupervised


def run_pipeline(df, target_column):
    problem_type = detect_problem_type_from_df(df, target_column)

    # --- Time Series ---
    if problem_type.startswith("Time Series"):
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                continue
        X_train, X_test, y_train, y_test = preprocess_time_series(df, target_column)
        full_eda(X_train)
        model, metrics = train_and_evaluate(problem_type, X_train, X_test, y_train, y_test)
        # encoders and scaler None for time series branch
        return model, problem_type, X_train.columns.tolist(), None, None, metrics

    # --- Classification ---
    elif problem_type == "Classification":
        X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_classification(df, target_column)
        full_eda(X_train)
        X_selected = select_features_model_based(X_train, y_train, problem_type)
        X_test_selected = X_test[X_selected.columns]

        # Fit a new scaler on the selected features so the scaler's feature names
        # match the columns that will be used at prediction time.
        selected_scaler = StandardScaler()
        X_selected_scaled = pd.DataFrame(
            selected_scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index,
        )
        X_test_selected_scaled = pd.DataFrame(
            selected_scaler.transform(X_test_selected),
            columns=X_test_selected.columns,
            index=X_test_selected.index,
        )

        model, metrics = train_and_evaluate(problem_type, X_selected_scaled, X_test_selected_scaled, y_train, y_test)
        return model, problem_type, X_selected.columns.tolist(), label_encoders, selected_scaler, metrics

    # --- Regression ---
    elif problem_type == "Regression":
        X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_regression(df, target_column)
        full_eda(X_train)
        X_selected = select_features_model_based(X_train, y_train, problem_type)
        X_test_selected = X_test[X_selected.columns]

        selected_scaler = StandardScaler()
        X_selected_scaled = pd.DataFrame(
            selected_scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index,
        )
        X_test_selected_scaled = pd.DataFrame(
            selected_scaler.transform(X_test_selected),
            columns=X_test_selected.columns,
            index=X_test_selected.index,
        )

        model, metrics = train_and_evaluate(problem_type, X_selected_scaled, X_test_selected_scaled, y_train, y_test)
        return model, problem_type, X_selected.columns.tolist(), label_encoders, selected_scaler, metrics

    # --- Unsupervised ---
    elif problem_type == "Unsupervised Learning":
        df_processed, label_encoders, scaler = preprocess_unsupervised(df)
        full_eda(df_processed)
        model, metrics = train_and_evaluate_unsupervised(df_processed)
        return model, problem_type, df_processed.columns.tolist(), label_encoders, scaler, metrics

    else:
        raise ValueError("Unsupported or unrecognized problem type.")



def detect_problem_type_from_df(df: pd.DataFrame, target_column: str) -> str:
    # mimic detect_problem_type() from file input
    df = df.copy()

    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

    # Convert date columns
    for col in df.columns:
        sample_value = str(df[col].iloc[0])
        if any(char.isdigit() for char in sample_value) and ("-" in sample_value or "/" in sample_value):
            try:
                df[col] = pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
            except:
                continue

    if target_column.lower() in ["", "none", "unsupervised", "na"]:
        return "Unsupervised Learning"

    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset.")

    target = df[target_column]
    date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]

    if date_cols:
        if df[date_cols[0]].is_monotonic_increasing:
            return f"Time Series Forecasting on '{target_column}'"

    if not is_numeric_dtype(target):
        return "Classification"

    unique_vals = target.nunique()

    if pd.api.types.is_integer_dtype(target) and unique_vals <= 20:
        return "Classification"

    if is_numeric_dtype(target) and unique_vals > 20:
        return "Regression"

    return "Unable to confidently determine problem type (check target values)"
