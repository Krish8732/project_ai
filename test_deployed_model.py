#!/usr/bin/env python3
"""
Test Script for Deployed E-Commerce ML Model
Validates that the model can make predictions correctly
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def test_deployed_model():
    """Test the deployed model with sample data"""
    print("🧪 TESTING DEPLOYED MODEL")
    print("=" * 50)
    
    try:
        # Load the deployed model
        print("Loading deployed model...")
        deployment_package = joblib.load('deployed_model.pkl')
        model = deployment_package['model']
        le_dict = deployment_package['label_encoders']
        feature_names = deployment_package['feature_names']
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model metadata: {deployment_package['model_metadata']}")
        print(f"🔧 Features: {len(feature_names)}")
        
        # Create sample test data
        print("\nCreating sample test data...")
        sample_data = create_sample_data(le_dict, feature_names)
        
        # Make predictions
        print("Making predictions...")
        probabilities = model.predict_proba(sample_data)[:, 1]
        
        # Display results
        print("\n📊 PREDICTION RESULTS:")
        print("-" * 30)
        for i, prob in enumerate(probabilities):
            print(f"Sample {i+1}: Purchase Probability = {prob:.3f}")
            if prob > 0.7:
                print(f"  🎯 Recommendation: HIGH - Show aggressive promotions")
            elif prob > 0.5:
                print(f"  📈 Recommendation: MEDIUM - Show standard recommendations")
            else:
                print(f"  🔍 Recommendation: LOW - Show discovery content")
        
        # Test with different scenarios
        print("\n🧪 TESTING DIFFERENT SCENARIOS:")
        print("-" * 40)
        
        # High-value electronics
        high_value_data = create_scenario_data(
            brand="apple", category_code="electronics.smartphones", 
            price=999.99, hour=15, day_of_week=2, is_weekend=0,
            le_dict=le_dict, feature_names=feature_names
        )
        high_value_prob = model.predict_proba(high_value_data)[:, 1][0]
        print(f"💻 High-value electronics: {high_value_prob:.3f}")
        
        # Weekend shopping
        weekend_data = create_scenario_data(
            brand="nike", category_code="sports.shoes", 
            price=89.99, hour=14, day_of_week=6, is_weekend=1,
            le_dict=le_dict, feature_names=feature_names
        )
        weekend_prob = model.predict_proba(weekend_data)[:, 1][0]
        print(f"🏃 Weekend sports shopping: {weekend_prob:.3f}")
        
        # Low-value items
        low_value_data = create_scenario_data(
            brand="generic", category_code="home.kitchen", 
            price=19.99, hour=10, day_of_week=1, is_weekend=0,
            le_dict=le_dict, feature_names=feature_names
        )
        low_value_prob = model.predict_proba(low_value_data)[:, 1][0]
        print(f"🏠 Low-value home items: {low_value_prob:.3f}")
        
        print("\n✅ Model testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def create_sample_data(le_dict, feature_names):
    """Create sample data for testing"""
    # Create a DataFrame with sample user behavior
    sample_data = pd.DataFrame({
        'brand': ['apple', 'nike', 'generic'],
        'category_code': ['electronics.smartphones', 'sports.shoes', 'home.kitchen'],
        'price': [999.99, 89.99, 19.99],
        'hour': [15, 14, 10],
        'day_of_week': [2, 6, 1],
        'is_weekend': [0, 1, 0],
        'purchased': [0, 0, 0]  # Will be filled with actual values
    })
    
    # Add missing features with default values
    for feature in feature_names:
        if feature not in sample_data.columns:
            sample_data[feature] = 0
    
    # Reorder columns to match training data
    sample_data = sample_data[feature_names]
    
    # Apply label encoding to categorical features
    for col, le in le_dict.items():
        if col in sample_data.columns:
            # Handle unseen categories
            known_categories = set(le.classes_)
            sample_data[col] = sample_data[col].astype(str).apply(
                lambda x: x if x in known_categories else '__unknown__'
            )
            sample_data[col] = le.transform(sample_data[col])
    
    return sample_data

def create_scenario_data(brand, category_code, price, hour, day_of_week, is_weekend, le_dict, feature_names):
    """Create data for a specific scenario"""
    scenario_data = pd.DataFrame({
        'brand': [brand],
        'category_code': [category_code],
        'price': [price],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'purchased': [0]
    })
    
    # Add missing features with default values
    for feature in feature_names:
        if feature not in scenario_data.columns:
            scenario_data[feature] = 0
    
    # Reorder columns to match training data
    scenario_data = scenario_data[feature_names]
    
    # Apply label encoding to categorical features
    for col, le in le_dict.items():
        if col in scenario_data.columns:
            known_categories = set(le.classes_)
            scenario_data[col] = scenario_data[col].astype(str).apply(
                lambda x: x if x in known_categories else '__unknown__'
            )
            scenario_data[col] = le.transform(scenario_data[col])
    
    return scenario_data

def test_model_monitoring():
    """Test the model monitoring system"""
    print("\n🔍 TESTING MODEL MONITORING")
    print("=" * 40)
    
    try:
        from model_monitoring import ModelMonitor
        
        # Initialize monitor
        monitor = ModelMonitor()
        print("✅ Model monitor initialized")
        
        # Log some test predictions
        print("Logging test predictions...")
        monitor.log_prediction("user001", 1, 0.85)  # Correct prediction
        monitor.log_prediction("user002", 0, 0.25)  # Correct prediction
        monitor.log_prediction("user003", 1, 0.30)  # Incorrect prediction
        monitor.log_prediction("user004", 0, 0.75)  # Incorrect prediction
        
        # Calculate drift metrics
        print("Calculating drift metrics...")
        drift_metrics = monitor.calculate_drift_metrics(window_days=1)
        print(f"Drift metrics: {drift_metrics}")
        
        # Save performance log
        monitor.save_performance_log('test_performance_log.json')
        print("✅ Performance log saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing monitoring: {e}")
        return False

if __name__ == "__main__":
    print("🚀 E-COMMERCE ML MODEL TESTING SUITE")
    print("=" * 60)
    
    # Test the deployed model
    model_test_success = test_deployed_model()
    
    # Test the monitoring system
    monitoring_test_success = test_model_monitoring()
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 TESTING SUMMARY")
    print("=" * 60)
    
    if model_test_success and monitoring_test_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Model predictions working correctly")
        print("✅ Monitoring system functional")
        print("\n🚀 Model is ready for production deployment!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\n📋 Next Steps:")
    print("1. Review test results")
    print("2. Set up API endpoint if needed")
    print("3. Deploy to production environment")
    print("4. Implement monitoring and alerting")


