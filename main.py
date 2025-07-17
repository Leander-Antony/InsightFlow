from prob_type import detect_problem_type
from preprocess import (
    preprocess_regression,
    preprocess_classification,
    preprocess_time_series,
    preprocess_unsupervised,
)
from feature_selection import select_features_statistical, select_features_model_based
from eda_utils import full_eda
import pandas as pd
from model_training import train_and_evaluate

def main():
    file_path = "datasets/test_linreg.csv"
    target_column = "belly"

    problem_type = detect_problem_type(file_path, target_column)
    print(f"Detected Problem Type: {problem_type}")

    df = pd.read_csv(file_path)

    if problem_type.startswith("Time Series"):
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                continue
        X_train, X_test, y_train, y_test = preprocess_time_series(df, target_column)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        model = train_and_evaluate(problem_type, X_train, X_test, y_train, y_test)

    elif problem_type == "Classification":
        X_train, X_test, y_train, y_test = preprocess_classification(df, target_column)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        full_eda(X_train)

        X_selected_model = select_features_model_based(X_train, y_train, problem_type)
        print(f"X_selected_model shape: {X_selected_model.shape}")

        # Align test set
        selected_columns = X_selected_model.columns
        X_test_selected = X_test[selected_columns]

        model = train_and_evaluate(problem_type, X_selected_model, X_test_selected, y_train, y_test)

    elif problem_type == "Regression":
        X_train, X_test, y_train, y_test = preprocess_regression(df, target_column)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        full_eda(X_train)

        X_selected_model = select_features_model_based(X_train, y_train, problem_type)
        print(f"X_selected_model shape: {X_selected_model.shape}")

        # Align test set
        selected_columns = X_selected_model.columns
        X_test_selected = X_test[selected_columns]

        model = train_and_evaluate(problem_type, X_selected_model, X_test_selected, y_train, y_test)

    elif problem_type == "Unsupervised Learning":
        df_processed = preprocess_unsupervised(df)
        print("Preprocessing completed successfully for Unsupervised Learning.")
        print(f"Processed Data Shape: {df_processed.shape}")
        full_eda(df_processed)

        # Dedicated unsupervised function instead of train_and_evaluate
        from model_training import train_and_evaluate_unsupervised
        model = train_and_evaluate_unsupervised(df_processed)


    else:
        print("Unsupported problem type for preprocessing.")
        return

    print("Preprocessing and training completed successfully.")

if __name__ == "__main__":
    main()
