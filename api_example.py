
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="E-Commerce Purchase Prediction API")

# Load deployed model
deployment_package = joblib.load('deployed_model.pkl')
model = deployment_package['model']

def preprocess_user_data(user_data, le_dict, feature_names):
    """Preprocess user data to match training format"""
    try:
        # Create a copy to avoid warnings
        user_data = user_data.copy()
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in user_data.columns:
                user_data[feature] = 0
        
        # Reorder columns to match training data
        user_data = user_data[feature_names]
        
        # Apply label encoding to categorical features
        for col, le in le_dict.items():
            if col in user_data.columns:
                # Handle unseen categories
                known_categories = set(le.classes_)
                user_data.loc[:, col] = user_data[col].astype(str).apply(
                    lambda x: x if x in known_categories else '__unknown__'
                )
                user_data.loc[:, col] = le.transform(user_data[col])
        
        return user_data
        
    except Exception as e:
        print(f"Error preprocessing user data: {e}")
        return None

class UserData(BaseModel):
    user_id: str
    session_id: str
    product_id: str
    category_code: str
    brand: str
    price: float
    hour: int
    day_of_week: int
    is_weekend: int
    # Add other features as needed

@app.post("/predict_purchase")
async def predict_purchase(user_data: UserData):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([user_data.model_dump()])
        
        # Preprocess and predict
        processed_data = preprocess_user_data(df, deployment_package['label_encoders'], deployment_package['feature_names'])
        probability = model.predict_proba(processed_data)[:, 1][0]
        
        return {
            "user_id": user_data.user_id,
            "purchase_probability": float(probability),
            "recommendation": "high" if probability > 0.7 else "medium" if probability > 0.5 else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
