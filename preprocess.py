# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import category_encoders as ce

# === COMMON UTILS ===
def separate_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def encode_categoricals_target(X, y, encoders=None, task='classification'):
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            unique_vals = X[col].nunique()
            # If high cardinality, use TargetEncoder. Else use OrdinalEncoder.
            if unique_vals > 5 and y is not None:
                te = ce.TargetEncoder()
                X[col] = te.fit_transform(X[col], y)
                encoders[col] = te
            else:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X[[col]] = oe.fit_transform(X[[col]])
                encoders[col] = oe
    else:
        for col in cat_cols:
            if col in encoders:
                encoder = encoders[col]
                if isinstance(encoder, ce.TargetEncoder):
                    # target encoder transform
                    X[col] = encoder.transform(X[col])
                else:
                    X[[col]] = encoder.transform(X[[col]])
    return X, encoders

def scale_features(X):
    # RobustScaler is less sensitive to outliers than StandardScaler
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler

# === REGRESSION ===
def preprocess_regression(df, target_column):
    X, y = separate_features_target(df, target_column)

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    # Impute numeric columns
    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='median') # Median is robust to outliers
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    # Impute categorical columns
    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

    # Ensure y is numeric
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = pd.to_numeric(y, errors='coerce')
        y = y.fillna(y.median())

    X, label_encoders = encode_categoricals_target(X, y, task='regression')
    X, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders, scaler

# === CLASSIFICATION ===
def preprocess_classification(df, target_column):
    X, y = separate_features_target(df, target_column)

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='median')
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

    # Target encoding needs numeric y
    is_y_categorical = y.dtype == 'object' or y.dtype.name == 'category'
    y_encoded = y
    if is_y_categorical:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

    X, label_encoders = encode_categoricals_target(X, pd.Series(y_encoded), task='classification')
    X, scaler = scale_features(X)

    # Use SMOTE to handle class imbalance if possible
    # We only apply SMOTE to the training set to prevent data leakage!
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Try SMOTE (will fail if a class has fewer than k_neighbors samples)
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train, y_train = X_train_resampled, y_train_resampled
    except Exception as e:
        print(f"Skipping SMOTE due to error (likely too few samples in a minority class): {e}")

    return X_train, X_test, y_train, y_test, label_encoders, scaler

# === TIME SERIES ===
def preprocess_time_series(df, target_column, date_column=None, n_lags=3):
    if date_column is None:
        date_candidates = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
        if not date_candidates:
            raise ValueError("No datetime column found in DataFrame. Please specify date_column manually.")
        date_column = date_candidates[0]

    df = df.sort_values(by=date_column)

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

    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy='median')
        df[num_cols] = imp_num.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        imp_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    # Unsupervised doesn't have a target, so we can't use TargetEncoder, fallback to ordinal
    cat_cols_list = df.select_dtypes(include=['object', 'category']).columns.tolist()
    label_encoders = {}
    for col in cat_cols_list:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[[col]] = oe.fit_transform(df[[col]])
        label_encoders[col] = oe

    df, scaler = scale_features(df)

    return df, label_encoders, scaler
