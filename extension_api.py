from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from deploy_model import preprocess_user_data

app = FastAPI(title="Extension Integration API")

# Allow requests from the browser extension (and localhost testing tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(path: str = "deployed_model.pkl"):
    try:
        pkg = joblib.load(path)
        return pkg
    except Exception as e:
        raise RuntimeError(f"Failed to load deployed model: {e}")


deployment_package = None


@app.on_event("startup")
def startup_event():
    global deployment_package
    deployment_package = load_model('deployed_model.pkl')
    print("✅ Loaded deployed_model.pkl for extension integration")


@app.post("/predict_event")
async def predict_event(payload: dict = Body(...)):
    """Accept a JSON payload of event features from the extension and return a prediction.

    Expected payload is a flat map of feature name -> value. Missing features are filled with 0.
    The endpoint will attempt to reuse the project's preprocessing (label encoders + feature ordering).
    """
    try:
        if deployment_package is None:
            raise RuntimeError("Model not loaded on server")

        feature_names = deployment_package['feature_names']
        le_dict = deployment_package['label_encoders']

        # Convert to DataFrame (single row)
        df = pd.DataFrame([payload])

        # Preprocess using shared preprocessing helper
        processed = preprocess_user_data(df, le_dict, feature_names)
        if processed is None:
            raise RuntimeError("Preprocessing returned None")

        model = deployment_package['model']
        prob = float(model.predict_proba(processed)[:, 1][0])

        # More granular recommendation levels with diverse colors
        if prob >= 0.9:
            recommendation = "very_high"
        elif prob >= 0.7:
            recommendation = "high"
        elif prob >= 0.5:
            recommendation = "medium_high"
        elif prob >= 0.3:
            recommendation = "medium"
        elif prob >= 0.1:
            recommendation = "low"
        else:
            recommendation = "very_low"

        return {
            "purchase_probability": prob,
            "recommendation": recommendation,
            "feature_count": len(feature_names)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
