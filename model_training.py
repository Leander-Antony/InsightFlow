import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    classification_report,
    silhouette_score
)
from sklearn.cluster import KMeans


def train_and_evaluate_unsupervised(X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_
    score = silhouette_score(X, labels)

    print(f"Silhouette Score: {score:.4f}")
    return kmeans


def train_and_evaluate(problem_type, X_train, X_test, y_train=None, y_test=None):
    if problem_type.startswith("Time Series") or problem_type == "Regression":
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Regression Metrics:")
        print(f"→ MSE: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"→ MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"→ R²: {r2_score(y_test, y_pred):.4f}")

    elif problem_type == "Classification":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Classification Metrics:")
        print(f"→ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"→ F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"→ Classification Report:\n{classification_report(y_test, y_pred)}")

    elif problem_type == "Unsupervised Learning":
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X_train)
        labels = model.labels_
        score = silhouette_score(X_train, labels)

        print(f"Unsupervised Learning Metrics:")
        print(f"→ Cluster counts:\n{pd.Series(labels).value_counts()}")
        print(f"→ Silhouette Score: {score:.4f}")

    else:
        print("Unsupported problem type for training.")
        return None

    return model
