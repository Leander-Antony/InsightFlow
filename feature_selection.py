import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def select_features_statistical(X, y, problem_type, k=5):
    if problem_type == "Classification":
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    elif problem_type == "Regression":
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    else:
        raise ValueError("Statistical feature selection only works for supervised learning.")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"\nTop {len(selected_features)} features (Statistical): {list(selected_features)}")
    return X[selected_features]

def select_features_model_based(X, y, problem_type):
    if problem_type == "Classification":
        model = RandomForestClassifier(random_state=42)
    elif problem_type == "Regression":
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Model-based feature selection only works for supervised learning.")
    
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances (Model-Based):")
    print(feature_importances.head(10))

    # Select top 5
    top_features = feature_importances.head(5).index
    return X[top_features]
