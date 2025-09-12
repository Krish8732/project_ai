#!/usr/bin/env python3
"""
Debug script to understand why model predictions are very low
"""

import joblib
import pandas as pd
import numpy as np

def debug_model_predictions():
    """Debug the model predictions to understand the issue"""
    print("🔍 DEBUGGING MODEL PREDICTIONS")
    print("=" * 50)
    
    try:
        # Load the deployed model
        deployment_package = joblib.load('deployed_model.pkl')
        model = deployment_package['model']
        le_dict = deployment_package['label_encoders']
        feature_names = deployment_package['feature_names']
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model metadata: {deployment_package['model_metadata']}")
        print(f"🔧 Features: {len(feature_names)}")
        print(f"📋 Feature names: {feature_names}")
        
        # Load the original training data to see what features look like
        print("\n📊 ANALYZING TRAINING DATA FEATURES:")
        print("-" * 40)
        
        X_train, X_val, X_test, y_train, y_val, y_test, le_dict_orig = joblib.load('mm_split.pkl')
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training data columns: {list(X_train.columns)}")
        
        # Check feature distributions
        print("\n📈 FEATURE DISTRIBUTIONS:")
        print("-" * 30)
        
        for col in X_train.columns[:10]:  # Show first 10 features
            if X_train[col].dtype in ['int64', 'float64']:
                print(f"{col}: mean={X_train[col].mean():.3f}, std={X_train[col].std():.3f}, min={X_train[col].min():.3f}, max={X_train[col].max():.3f}")
            else:
                print(f"{col}: unique_values={X_train[col].nunique()}")
        
        # Test with actual training data sample
        print("\n🧪 TESTING WITH TRAINING DATA SAMPLE:")
        print("-" * 40)
        
        sample_idx = np.random.randint(0, len(X_train), 5)
        for i, idx in enumerate(sample_idx):
            sample_data = X_train.iloc[idx:idx+1]
            actual_label = y_train.iloc[idx]
            
            # Make prediction
            prob = model.predict_proba(sample_data)[:, 1][0]
            print(f"Sample {i+1}: Actual={actual_label}, Predicted_Prob={prob:.6f}")
        
        # Test with our API test case
        print("\n🧪 TESTING API TEST CASE:")
        print("-" * 30)
        
        api_test_data = pd.DataFrame({
            'brand': ['apple'],
            'category_code': ['electronics.smartphones'],
            'price': [999.99],
            'hour': [15],
            'day_of_week': [3],
            'is_weekend': [0],
            'purchased': [0]
        })
        
        # Add missing features with default values
        for feature in feature_names:
            if feature not in api_test_data.columns:
                api_test_data[feature] = 0
        
        # Reorder columns to match training data
        api_test_data = api_test_data[feature_names]
        
        # Apply label encoding to categorical features
        for col, le in le_dict.items():
            if col in api_test_data.columns:
                known_categories = set(le.classes_)
                api_test_data[col] = api_test_data[col].astype(str).apply(
                    lambda x: x if x in known_categories else '__unknown__'
                )
                api_test_data[col] = le.transform(api_test_data[col])
        
        # Make prediction
        api_prob = model.predict_proba(api_test_data)[:, 1][0]
        print(f"API test case: Predicted_Prob={api_prob:.6f}")
        
        # Check what the model sees
        print(f"\n🔍 WHAT THE MODEL SEES:")
        print(f"Input shape: {api_test_data.shape}")
        print(f"Input columns: {list(api_test_data.columns)}")
        print(f"First few values: {api_test_data.iloc[0, :5].values}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error debugging model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_model_predictions()


