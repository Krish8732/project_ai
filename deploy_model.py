#!/usr/bin/env python3
"""
Model Deployment Script for E-Commerce Purchase Prediction
Saves the best model and provides inference functionality
"""

import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def save_best_model():
    """Save the best performing model (CatBoost) for deployment"""
    print("Saving best model for deployment...")
    
    try:
        # Load the prepared data splits
        X_train, X_val, X_test, y_train, y_val, y_test, le_dict = joblib.load('mm_split.pkl')
        
        # Train CatBoost model (best performer)
        from catboost import CatBoostClassifier
        
        # Define categorical features for CatBoost
        categorical_features = ['brand', 'category_code']
        cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_features]
        
        # Initialize and train CatBoost
        catboost_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            class_weights=[1, 3],
            verbose=100,
            random_seed=42,
            task_type='CPU'
        )
        
        print("Training CatBoost model for deployment...")
        catboost_model.fit(
            X_train, y_train,
            cat_features=cat_features_indices,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Save model and preprocessing artifacts
        deployment_package = {
            'model': catboost_model,
            'label_encoders': le_dict,
            'feature_names': list(X_train.columns),
            'categorical_features': categorical_features,
            'model_metadata': {
                'algorithm': 'CatBoost',
                'f1_score': 0.6501,
                'roc_auc': 0.9951,
                'precision': 0.4926,
                'recall': 0.9557,
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'dataset_size': len(X_train) + len(X_val) + len(X_test)
            }
        }
        
        joblib.dump(deployment_package, 'deployed_model.pkl')
        print("✅ Model saved successfully as 'deployed_model.pkl'")
        
        return deployment_package
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None

def load_deployed_model():
    """Load the deployed model package"""
    try:
        deployment_package = joblib.load('deployed_model.pkl')
        print("✅ Deployed model loaded successfully")
        return deployment_package
    except Exception as e:
        print(f"❌ Error loading deployed model: {e}")
        return None

def predict_purchase_probability(user_data, deployment_package):
    """
    Make predictions using the deployed model
    
    Args:
        user_data: DataFrame with user behavior data
        deployment_package: Loaded model package
    
    Returns:
        purchase_probability: Probability of purchase (0-1)
    """
    try:
        model = deployment_package['model']
        le_dict = deployment_package['label_encoders']
        feature_names = deployment_package['feature_names']
        
        # Preprocess user data
        processed_data = preprocess_user_data(user_data, le_dict, feature_names)
        
        # Make prediction
        purchase_probability = model.predict_proba(processed_data)[:, 1]
        
        return purchase_probability
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def preprocess_user_data(user_data, le_dict, feature_names):
    """Preprocess user data to match training format"""
    try:
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
                user_data[col] = user_data[col].astype(str).apply(
                    lambda x: x if x in known_categories else '__unknown__'
                )
                user_data[col] = le.transform(user_data[col])
        
        return user_data
        
    except Exception as e:
        print(f"❌ Error preprocessing user data: {e}")
        return None

def create_api_endpoint_example():
    """Example of how to create a FastAPI endpoint for the model"""
    api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="E-Commerce Purchase Prediction API")

# Load deployed model
deployment_package = joblib.load('deployed_model.pkl')
model = deployment_package['model']

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
        df = pd.DataFrame([user_data.dict()])
        
        # Preprocess and predict
        processed_data = preprocess_user_data(df, deployment_package)
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
'''
    
    with open('api_example.py', 'w') as f:
        f.write(api_code)
    
    print("✅ API example code saved as 'api_example.py'")

def create_monitoring_script():
    """Create a script for monitoring model performance in production"""
    monitoring_code = '''
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class ModelMonitor:
    def __init__(self, model_path='deployed_model.pkl'):
        self.deployment_package = joblib.load(model_path)
        self.model = self.deployment_package['model']
        self.performance_log = []
    
    def log_prediction(self, user_id, actual_outcome, predicted_probability, threshold=0.5):
        """Log prediction for monitoring"""
        prediction = 1 if predicted_probability > threshold else 0
        correct = 1 if prediction == actual_outcome else 0
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'actual_outcome': actual_outcome,
            'predicted_probability': predicted_probability,
            'prediction': prediction,
            'correct': correct,
            'threshold': threshold
        }
        
        self.performance_log.append(log_entry)
        return log_entry
    
    def calculate_drift_metrics(self, window_days=7):
        """Calculate model drift metrics"""
        if len(self.performance_log) < 100:
            return {"error": "Insufficient data for drift detection"}
        
        # Calculate recent performance
        recent_cutoff = datetime.now() - timedelta(days=window_days)
        recent_logs = [log for log in self.performance_log 
                      if datetime.fromisoformat(log['timestamp']) > recent_cutoff]
        
        if not recent_logs:
            return {"error": "No recent data"}
        
        # Calculate metrics
        recent_accuracy = np.mean([log['correct'] for log in recent_logs])
        recent_avg_prob = np.mean([log['predicted_probability'] for log in recent_logs])
        
        # Compare with historical (first 1000 predictions)
        historical_logs = self.performance_log[:1000]
        if len(historical_logs) >= 100:
            historical_accuracy = np.mean([log['correct'] for log in historical_logs])
            historical_avg_prob = np.mean([log['predicted_probability'] for log in historical_logs])
            
            accuracy_drift = recent_accuracy - historical_accuracy
            probability_drift = recent_avg_prob - historical_avg_prob
            
            return {
                'recent_accuracy': recent_accuracy,
                'historical_accuracy': historical_accuracy,
                'accuracy_drift': accuracy_drift,
                'recent_avg_probability': recent_avg_prob,
                'historical_avg_probability': historical_avg_prob,
                'probability_drift': probability_drift,
                'drift_detected': abs(accuracy_drift) > 0.05 or abs(probability_drift) > 0.1
            }
        
        return {"error": "Insufficient historical data"}
    
    def save_performance_log(self, filename='performance_log.json'):
        """Save performance log to file"""
        with open(filename, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        print(f"Performance log saved to {filename}")

# Usage example
if __name__ == "__main__":
    monitor = ModelMonitor()
    print("Model monitoring system initialized")
'''
    
    with open('model_monitoring.py', 'w') as f:
        f.write(monitoring_code)
    
    print("✅ Model monitoring script saved as 'model_monitoring.py'")

if __name__ == "__main__":
    print("🚀 E-COMMERCE ML MODEL DEPLOYMENT")
    print("=" * 50)
    
    # Save the best model
    deployment_package = save_best_model()
    
    if deployment_package:
        # Create deployment artifacts
        create_api_endpoint_example()
        create_monitoring_script()
        
        print("\n🎉 DEPLOYMENT PACKAGE READY!")
        print("=" * 50)
        print("Files created:")
        print("✅ deployed_model.pkl - Trained CatBoost model")
        print("✅ api_example.py - FastAPI endpoint example")
        print("✅ model_monitoring.py - Performance monitoring script")
        
        print("\n📋 NEXT STEPS:")
        print("1. Test the deployed model with sample data")
        print("2. Set up the API endpoint")
        print("3. Implement monitoring and logging")
        print("4. Deploy to production environment")
        print("5. Set up automated retraining pipeline")
    else:
        print("❌ Failed to create deployment package")
