# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# === COMMON UTILS ===
def separate_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def encode_categoricals(X):
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    return X, label_encoders

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler

# === REGRESSION ===
def preprocess_regression(df, target_column):
    X, y = separate_features_target(df, target_column)

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    # Impute numeric columns
    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='mean')
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    # Impute categorical columns
    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

    X, _ = encode_categoricals(X)
    X, _ = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# === CLASSIFICATION ===
def preprocess_classification(df, target_column):
    X, y = separate_features_target(df, target_column)

    # Impute missing values
    imp = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    X, _ = encode_categoricals(X)
    X, _ = scale_features(X)

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# === TIME SERIES ===
def preprocess_time_series(df, target_column, date_column=None, n_lags=3):
    # Fallback: look for datetime columns or raise an error
    if date_column is None:
        date_candidates = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
        if not date_candidates:
            raise ValueError("No datetime column found in DataFrame. Please specify date_column manually.")
        date_column = date_candidates[0]

    df = df.sort_values(by=date_column)

    # Lag features
    for i in range(1, n_lags + 1):
        df[f'{target_column}_lag{i}'] = df[target_column].shift(i)

    df = df.dropna()

    X = df[[f'{target_column}_lag{i}' for i in range(1, n_lags + 1)]]
    y = df[target_column]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# === UNSUPERVISED ===
def preprocess_unsupervised(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # Impute numeric columns
    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='mean')
        df[num_cols] = imp_num.fit_transform(df[num_cols])

    # Impute categorical columns
    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    df, _ = encode_categoricals(df)
    df, _ = scale_features(df)

    return df

