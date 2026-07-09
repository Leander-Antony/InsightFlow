# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
from main import run_pipeline
from preprocess import encode_categoricals, scale_features

# === Page Config & Custom CSS ===
st.set_page_config(page_title="AutoML Pipeline", layout="wide", page_icon="🔍")

st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stForm {
        border-radius: 12px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 AutoML Pipeline")
st.markdown("Automated Problem Detection & Model Training")

# === Session State Initialization ===
for key in ['model', 'problem_type', 'feature_cols', 'df', 'encoders', 'scaler', 'trained', 'metrics', 'target']:
    if key not in st.session_state:
        st.session_state[key] = None

# === Sidebar for Configuration ===
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Upload your dataset and select the target variable.")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
    
    if uploaded_file:
        # Check if file changed to reset state
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.last_file = uploaded_file.name
            st.session_state.trained = False
            st.session_state.feature_cols = None
            st.session_state.model = None
            st.session_state.target = None
            
        columns = st.session_state.df.columns.tolist()
        
        # Target column selection
        selected_target = st.selectbox("🎯 Target Column", [""] + columns, index=([""] + columns).index(st.session_state.target) if st.session_state.target in ([""] + columns) else 0, help="Leave empty for Unsupervised Learning")
        
        if st.session_state.target != selected_target:
            st.session_state.target = selected_target
            st.session_state.trained = False
            st.session_state.feature_cols = None
            st.session_state.model = None
        
        run_button = st.button("🚀 Run AutoML", type="primary")
    else:
        run_button = False

# === Main Dashboard ===
if st.session_state.df is not None:
    # Data Preview
    with st.expander("📊 Dataset Preview", expanded=True):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        st.caption(f"Shape: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")

    if run_button:
        with st.spinner("Training... please wait..."):
            try:
                result = run_pipeline(st.session_state.df, st.session_state.target)
                model, problem_type, feature_cols, encoders, scaler, metrics = result

                st.session_state.model = model
                st.session_state.problem_type = problem_type
                st.session_state.feature_cols = feature_cols
                st.session_state.encoders = encoders
                st.session_state.scaler = scaler
                st.session_state.metrics = metrics
                st.session_state.trained = True

            except Exception as e:
                st.error(f"❌ Error during training: {e}")

    # Results & Metrics
    if st.session_state.trained:
        st.success(f"✅ Model trained successfully! Detected Problem Type: **{st.session_state.problem_type}**")
        
        if st.session_state.metrics:
            st.subheader("📈 Training Metrics")
            numeric = {k: v for k, v in st.session_state.metrics.items() if isinstance(v, (int, float))}
            other = {k: v for k, v in st.session_state.metrics.items() if k not in numeric}
            
            if numeric:
                cols = st.columns(len(numeric))
                for idx, (k, v) in enumerate(numeric.items()):
                    cols[idx].metric(label=k.replace('_', ' ').title(), value=round(v, 4))
            if other:
                with st.expander("Additional Metrics"):
                    st.json(other)

        # Test Model
        st.divider()
        st.subheader("🧪 Test the Trained Model")
        st.markdown("Enter feature values below to get a prediction.")
        
        with st.form("prediction_form"):
            # Create a grid for inputs (3 columns per row)
            input_cols = st.columns(3)
            test_input = {}
            
            for idx, feature in enumerate(st.session_state.feature_cols):
                col = input_cols[idx % 3]
                with col:
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
            
            st.write("") # Spacer
            submit_pred = st.form_submit_button("🧾 Generate Prediction", type="primary")

        if submit_pred:
            with st.spinner("Predicting..."):
                try:
                    test_df = pd.DataFrame([test_input])

                    # --- Encode using trained encoders ---
                    if st.session_state.encoders and isinstance(st.session_state.encoders, dict):
                        for col, encoder in st.session_state.encoders.items():
                            if col in test_df.columns:
                                try:
                                    transformed = encoder.transform(test_df[[col]])
                                except Exception:
                                    transformed = encoder.transform(test_df[col].astype(str).values.reshape(-1, 1))
                                if hasattr(transformed, "shape") and transformed.shape[1] == 1:
                                    test_df[col] = transformed.ravel()
                                else:
                                    test_df[col] = transformed

                    # Add missing cols
                    for col in st.session_state.feature_cols:
                        if col not in test_df.columns:
                            test_df[col] = 0

                    test_df = test_df[st.session_state.feature_cols]

                    # Scale
                    if st.session_state.scaler:
                        test_df = pd.DataFrame(
                            st.session_state.scaler.transform(test_df),
                            columns=st.session_state.feature_cols
                        )

                    # Predict
                    prediction = st.session_state.model.predict(test_df)
                    st.success(f"### 📣 Prediction: **{prediction[0]}**")

                    # Confidence
                    model = st.session_state.model
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(test_df)
                        top_conf = probs[0].max()
                        st.info(f"**Confidence (probability):** {top_conf:.2%}")
                    elif hasattr(model, "estimators_"):
                        all_preds = np.vstack([est.predict(test_df) for est in model.estimators_])
                        std = all_preds.std(axis=0)[0]
                        st.info(f"**Prediction uncertainty (std):** {std:.4f}")

                except Exception as pred_e:
                    st.error(f"Prediction failed: {pred_e}")
else:
    st.info("👈 Please upload a dataset from the sidebar to get started.")

