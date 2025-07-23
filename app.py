import streamlit as st
import pandas as pd
from main import run_pipeline
from preprocess import encode_categoricals, scale_features

st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.title("🔍 AutoML: Automated Problem Detection & Model Training")

# === Session State Initialization ===
for key in ['model', 'problem_type', 'feature_cols', 'df', 'encoders', 'scaler', 'trained']:
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
                result = run_pipeline(st.session_state.df, target)
                if len(result) == 3:  # Time Series
                    model, problem_type, feature_cols = result
                    encoders, scaler = None, None
                else:
                    model, problem_type, feature_cols, encoders, scaler = result

                st.session_state.model = model
                st.session_state.problem_type = problem_type
                st.session_state.feature_cols = feature_cols
                st.session_state.encoders = encoders
                st.session_state.scaler = scaler
                st.session_state.trained = True

                st.success(f"✅ Model trained successfully! Problem Type: **{problem_type}**")
                st.write(model)

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

            # Encode using trained encoders
            if st.session_state.encoders:
                for col, encoder in st.session_state.encoders.items():
                    if col in test_df.columns:
                        test_df[col] = encoder.transform(test_df[[col]])

            # Add missing columns that were in training but not in test
            for col in st.session_state.feature_cols:
                if col not in test_df.columns:
                    test_df[col] = 0  # default value or use training median if you want

            # Reorder columns to match training order
            test_df = test_df[st.session_state.feature_cols]

            # Apply trained scaler
            if st.session_state.scaler:
                test_df = pd.DataFrame(
                    st.session_state.scaler.transform(test_df),
                    columns=st.session_state.feature_cols
                )

            prediction = st.session_state.model.predict(test_df)
            st.success(f"📣 Prediction: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

