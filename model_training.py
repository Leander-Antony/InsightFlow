import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import shap

# Disable Optuna logging to avoid cluttering the output
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 5 # Small number of trials for the web demo to remain responsive

def train_and_evaluate_unsupervised(X):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    metrics = {
        "silhouette_score": float(score),
        "best_model": "K-Means (k=3)"
    }
    return kmeans, metrics, None

# === OPTUNA OBJECTIVES ===
def objective_rf_clf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score

def objective_xgb_clf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score

def objective_lgb_clf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score

def objective_rf_reg(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': 42
    }
    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return score

def objective_xgb_reg(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return score

def objective_lgb_reg(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
    return score

# === TRAINING LOGIC ===
def train_and_evaluate(problem_type, X_train, X_test, y_train=None, y_test=None):
    metrics = {}
    leaderboard = {}

    if problem_type == "Unsupervised Learning":
        return train_and_evaluate_unsupervised(X_train)

    if problem_type == "Classification":
        print("Tuning Classification Models with Optuna...")
        
        # 1. Random Forest
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(lambda t: objective_rf_clf(t, X_train, y_train), n_trials=N_TRIALS)
        rf_best = RandomForestClassifier(**study_rf.best_params, random_state=42)
        
        # 2. XGBoost
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda t: objective_xgb_clf(t, X_train, y_train), n_trials=N_TRIALS)
        xgb_params = study_xgb.best_params
        xgb_params['use_label_encoder'] = False
        xgb_params['eval_metric'] = 'logloss'
        xgb_best = XGBClassifier(**xgb_params, random_state=42)
        
        # 3. LightGBM
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(lambda t: objective_lgb_clf(t, X_train, y_train), n_trials=N_TRIALS)
        lgb_params = study_lgb.best_params
        lgb_params['verbose'] = -1
        lgb_best = LGBMClassifier(**lgb_params, random_state=42)

        # 4. Logistic Regression (Baseline)
        log_reg = LogisticRegression(max_iter=1000, random_state=42)

        models = {
            "Logistic Regression": log_reg,
            "Tuned Random Forest": rf_best,
            "Tuned XGBoost": xgb_best,
            "Tuned LightGBM": lgb_best
        }

        # Stacking Ensemble
        print("Building Stacking Classifier...")
        estimators = [
            ('xgb', xgb_best),
            ('lgb', lgb_best),
            ('rf', rf_best)
        ]
        stacking_clf = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(),
            cv=3
        )
        models["Advanced Stacking Ensemble"] = stacking_clf

        # Evaluate all models for the leaderboard
        best_score = -np.inf
        best_model = None
        best_model_name = ""

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                leaderboard[name] = {"accuracy": float(acc), "f1": float(f1)}
                
                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Failed to train {name}: {e}")

        # Final Evaluation for best model
        y_pred = best_model.predict(X_test)
        metrics.update({
            "best_model": best_model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average='weighted')),
            "leaderboard": leaderboard
        })

    elif problem_type.startswith("Time Series") or problem_type == "Regression":
        print("Tuning Regression Models with Optuna...")
        
        # 1. Random Forest
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(lambda t: objective_rf_reg(t, X_train, y_train), n_trials=N_TRIALS)
        rf_best = RandomForestRegressor(**study_rf.best_params, random_state=42)
        
        # 2. XGBoost
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda t: objective_xgb_reg(t, X_train, y_train), n_trials=N_TRIALS)
        xgb_best = XGBRegressor(**study_xgb.best_params, random_state=42)
        
        # 3. LightGBM
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(lambda t: objective_lgb_reg(t, X_train, y_train), n_trials=N_TRIALS)
        lgb_params = study_lgb.best_params
        lgb_params['verbose'] = -1
        lgb_best = LGBMRegressor(**lgb_params, random_state=42)

        # 4. Linear Regression (Baseline)
        lin_reg = LinearRegression()

        models = {
            "Linear Regression": lin_reg,
            "Tuned Random Forest": rf_best,
            "Tuned XGBoost": xgb_best,
            "Tuned LightGBM": lgb_best
        }

        # Stacking Ensemble
        print("Building Stacking Regressor...")
        estimators = [
            ('xgb', xgb_best),
            ('lgb', lgb_best),
            ('rf', rf_best)
        ]
        stacking_reg = StackingRegressor(
            estimators=estimators, 
            final_estimator=Ridge(),
            cv=3
        )
        models["Advanced Stacking Ensemble"] = stacking_reg

        best_score = np.inf
        best_model = None
        best_model_name = ""

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                leaderboard[name] = {"mse": float(mse), "r2": float(r2)}
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Failed to train {name}: {e}")

        # Final Evaluation for best model
        y_pred = best_model.predict(X_test)
        metrics.update({
            "best_model": best_model_name,
            "mse": float(mean_squared_error(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "leaderboard": leaderboard
        })
    else:
        print("Unsupported problem type.")
        return None, metrics, None

    # Try to create a SHAP explainer
    # Note: SHAP can struggle directly with Stacking classifiers. 
    # If the best model is the ensemble, we use a KernelExplainer.
    shap_explainer = None
    try:
        if "Ensemble" in best_model_name:
            # We sample down the background data significantly for KernelExplainer to be fast
            background = shap.sample(X_train, 25)
            shap_explainer = shap.KernelExplainer(best_model.predict, background)
        elif best_model_name in ["Tuned Random Forest", "Tuned XGBoost", "Tuned LightGBM"]:
            shap_explainer = shap.TreeExplainer(best_model)
        elif best_model_name in ["Logistic Regression", "Linear Regression"]:
            shap_explainer = shap.LinearExplainer(best_model, X_train)
    except Exception as e:
        print(f"Could not create SHAP explainer: {e}")

    return best_model, metrics, shap_explainer
