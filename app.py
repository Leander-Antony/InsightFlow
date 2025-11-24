import streamlit as st
import pandas as pd
import numpy as np
from main import run_pipeline
from preprocess import encode_categoricals, scale_features

st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.title("🔍 AutoML: Automated Problem Detection & Model Training")

# === Session State Initialization ===
for key in ['model', 'problem_type', 'feature_cols', 'df', 'encoders', 'scaler', 'trained', 'metrics']:
    if key not in st.session_state:
        st.session_state[key] = None

# === File Upload ===
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Dataset")
    st.dataframe(st.session_state.df.head(10))

    columns = st.session_state.df.columns.tolist()
    target = st.selectbox("🎯 Select Target Column (Leave empty for Unsupervised)", [""] + columns)

    # Only train if not already trained or file has changed
    if st.button("🚀 Run AutoML"):
        with st.spinner("Training... please wait..."):
            try:
                # run_pipeline now returns a consistent 6-tuple:
                # (model, problem_type, feature_cols, encoders, scaler, metrics)
                result = run_pipeline(st.session_state.df, target)
                model, problem_type, feature_cols, encoders, scaler, metrics = result

                st.session_state.model = model
                st.session_state.problem_type = problem_type
                st.session_state.feature_cols = feature_cols
                st.session_state.encoders = encoders
                st.session_state.scaler = scaler
                st.session_state.metrics = metrics
                st.session_state.trained = True

                st.success(f"✅ Model trained successfully! Problem Type: **{problem_type}**")
                st.write(model)

                # Display metrics if available
                if metrics:
                    st.subheader("📈 Training metrics")
                    # Show numeric metrics as key: value
                    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    other = {k: v for k, v in metrics.items() if k not in numeric}
                    if numeric:
                        for k, v in numeric.items():
                            st.metric(label=k, value=round(v, 4))
                    if other:
                        st.write("Additional metrics:")
                        st.write(other)

            except Exception as e:
                st.error(f"❌ Error: {e}")

# === Prediction Interface (Only after training) ===
if st.session_state.trained and st.session_state.model and st.session_state.feature_cols:
    st.subheader("🧪 Test the Trained Model")

    test_input = {}
    for feature in st.session_state.feature_cols:
        if st.session_state.df[feature].dtype == 'object':
            test_input[feature] = st.selectbox(
                f"{feature} (categorical)", 
                st.session_state.df[feature].unique(),
                key=f"input_{feature}"
            )
        else:
            default_val = float(st.session_state.df[feature].mean())
            test_input[feature] = st.number_input(
                f"{feature}", 
                value=default_val, 
                key=f"input_{feature}"
            )

    if st.button("🧾 Predict"):
        try:
            test_df = pd.DataFrame([test_input])
            test_df = pd.DataFrame([test_input])

            # --- Encode using trained encoders (defensive) ---
            try:
                if st.session_state.encoders and isinstance(st.session_state.encoders, dict):
                    for col, encoder in st.session_state.encoders.items():
                        if col in test_df.columns:
                            # Some encoders expect 2D inputs; pass DataFrame slice
                            try:
                                transformed = encoder.transform(test_df[[col]])
                            except Exception:
                                # Try 1D transform for label-like encoders
                                transformed = encoder.transform(test_df[col].astype(str).values.reshape(-1, 1))
                            # If result is 2D, take flattened column
                            if hasattr(transformed, "shape") and transformed.shape[1] == 1:
                                test_df[col] = transformed.ravel()
                            else:
                                test_df[col] = transformed
            except Exception as enc_e:
                import traceback
                st.error("Encoding step failed. See traceback below:")
                st.text(traceback.format_exc())
                raise enc_e

            # Add missing columns that were in training but not in test
            for col in st.session_state.feature_cols:
                if col not in test_df.columns:
                    test_df[col] = 0  # default value or use training median if you want

            # Reorder columns to match training order
            test_df = test_df[st.session_state.feature_cols]

            # Apply trained scaler (defensive)
            try:
                if st.session_state.scaler:
                    test_df = pd.DataFrame(
                        st.session_state.scaler.transform(test_df),
                        columns=st.session_state.feature_cols
                    )
            except Exception as scale_e:
                import traceback
                st.error("Scaling step failed. See traceback below:")
                st.text(traceback.format_exc())
                raise scale_e

            # Show debug info (shape / dtypes) to help debug issues in Streamlit
            try:
                st.write("Test input shape:", test_df.shape)
                st.write(test_df.dtypes)
            except Exception:
                pass

            # Predict
            try:
                prediction = st.session_state.model.predict(test_df)
                st.success(f"📣 Prediction: **{prediction[0]}**")

                # Show confidence / uncertainty
                try:
                    model = st.session_state.model
                    # Classification: predict_proba
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(test_df)
                        top_conf = probs[0].max()
                        st.info(f"Confidence (predicted class probability): {top_conf:.2%}")
                    # Regression: if RandomForestRegressor, show std across estimators
                    elif hasattr(model, "estimators_"):
                        # collect predictions from individual estimators
                        all_preds = np.vstack([est.predict(test_df) for est in model.estimators_])
                        std = all_preds.std(axis=0)[0]
                        st.info(f"Prediction uncertainty (std across ensemble): {std:.4f}")
                    else:
                        st.info("No confidence information available for this model type.")
                except Exception:
                    # don't fail the whole predict when confidence calc fails
                    try:
                        import traceback
                        st.text(traceback.format_exc())
                    except Exception:
                        pass
            except Exception as pred_e:
                import traceback
                st.error("Prediction failed. See traceback below:")
                st.text(traceback.format_exc())
                raise pred_e
        except Exception as e:
            # final fallback - show full traceback
            import traceback
            st.error("An unexpected error occurred during prediction. See traceback below:")
            st.text(traceback.format_exc())
            st.error(f"Error summary: {e}")

