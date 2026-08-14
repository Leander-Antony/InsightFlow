from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uuid
import io
import json

from main import run_pipeline

app = FastAPI(title="InsightFlow AutoML API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the portfolio demo
demo_sessions = {}

class TrainRequest(BaseModel):
    session_id: str
    target_column: str

class PredictRequest(BaseModel):
    session_id: str
    features: dict

@app.get("/")
def read_root():
    return {"status": "InsightFlow API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    # Get data preview (first 5 rows)
    preview_df = df.head(5).fillna("")
    preview_data = preview_df.to_dict(orient='records')
    
    # Get column info for dynamic form generation and EDA
    column_info = {}
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) < 50: # Only send unique values for low cardinality
                column_info[col] = {"type": "categorical", "values": unique_vals, "missing": missing_count, "unique": unique_count}
            else:
                column_info[col] = {"type": "text", "values": [], "missing": missing_count, "unique": unique_count}
        else:
            column_info[col] = {"type": "numeric", "mean": float(df[col].mean()) if not df[col].isna().all() else 0, "missing": missing_count, "unique": unique_count}

    session_id = str(uuid.uuid4())
    demo_sessions[session_id] = {
        "df": df,
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "column_info": column_info
    }
    
    return {
        "session_id": session_id,
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "preview_data": preview_data,
        "column_info": column_info
    }

@app.post("/train")
def train_model(req: TrainRequest):
    if req.session_id not in demo_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload the file again.")
    
    df = demo_sessions[req.session_id]["df"]
    
    if req.target_column not in df.columns and req.target_column != "":
        raise HTTPException(status_code=400, detail="Target column not found in dataset")
        
    target = req.target_column if req.target_column != "" else None
    
    try:
        # run_pipeline will return the trained model, metrics, etc.
        model, problem_type, feature_cols, encoders, scaler, metrics, shap_explainer = run_pipeline(df, target)
        
        demo_sessions[req.session_id].update({
            "model": model,
            "problem_type": problem_type,
            "feature_cols": feature_cols,
            "encoders": encoders,
            "scaler": scaler,
            "metrics": metrics,
            "shap_explainer": shap_explainer
        })
        
        return {
            "status": "success",
            "problem_type": problem_type,
            "metrics": metrics,
            "feature_cols": feature_cols
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
def predict(req: PredictRequest):
    if req.session_id not in demo_sessions or "model" not in demo_sessions[req.session_id]:
        raise HTTPException(status_code=404, detail="Session or trained model not found")
        
    session = demo_sessions[req.session_id]
    feature_cols = session["feature_cols"]
    encoders = session["encoders"]
    scaler = session["scaler"]
    model = session["model"]
    
    # Create DataFrame from input
    input_data = req.features
    test_df = pd.DataFrame([input_data])
    
    # Encode
    if encoders:
        for col, encoder in encoders.items():
            if col in test_df.columns:
                try:
                    transformed = encoder.transform(test_df[[col]])
                except:
                    transformed = encoder.transform(test_df[col].astype(str).values.reshape(-1, 1))
                if hasattr(transformed, "shape") and transformed.shape[1] == 1:
                    test_df[col] = transformed.ravel()
                else:
                    test_df[col] = transformed

    # Fill missing columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
            
    test_df = test_df[feature_cols]
    
    # Scale
    if scaler:
        test_df = pd.DataFrame(scaler.transform(test_df), columns=feature_cols)
        
    # Predict
    prediction = model.predict(test_df)
    
    # Calculate SHAP values
    shap_values_dict = {}
    if session.get("shap_explainer"):
        try:
            explainer = session["shap_explainer"]
            shap_vals = explainer.shap_values(test_df)
            
            # Formatting SHAP values for the frontend
            if isinstance(shap_vals, list):
                # For classification, taking the SHAP values of the predicted class
                class_idx = int(prediction[0]) if isinstance(prediction[0], (int, np.integer)) else 0
                if class_idx < len(shap_vals):
                    vals = shap_vals[class_idx][0]
                else:
                    vals = shap_vals[0][0]
            else:
                vals = shap_vals[0]
                
            shap_values_dict = {col: float(val) for col, val in zip(feature_cols, vals)}
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            
    result = {
        "prediction": float(prediction[0]) if isinstance(prediction[0], (np.number, float, int)) else str(prediction[0]),
        "shap_values": shap_values_dict
    }
    
    # Add probability if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test_df)[0]
        result["confidence"] = float(probs.max())
        
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
