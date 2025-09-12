#!/usr/bin/env python3
"""
Enhanced E-Commerce Purchase Prediction API
Generates realistic feature values for missing features
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import random

app = FastAPI(title="Enhanced E-Commerce Purchase Prediction API")

# Load deployed model
deployment_package = joblib.load('deployed_model.pkl')
model = deployment_package['model']
le_dict = deployment_package['label_encoders']
feature_names = deployment_package['feature_names']

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

def generate_realistic_features(user_data):
    """Generate realistic feature values based on input data"""
    try:
        # Create a copy to avoid warnings
        user_data = user_data.copy()
        
        # Generate realistic session-level features based on price and category
        price = user_data['price'].iloc[0]
        is_weekend = user_data['is_weekend'].iloc[0]
        hour = user_data['hour'].iloc[0]
        
        # Session behavior features (more realistic)
        if price > 500:  # High-value items
            total_events = random.randint(3, 8)  # More browsing for expensive items
            unique_products = random.randint(2, 5)
            session_duration = random.randint(15, 45)  # Longer sessions
        elif price > 100:  # Medium-value items
            total_events = random.randint(2, 5)
            unique_products = random.randint(1, 3)
            session_duration = random.randint(10, 25)
        else:  # Low-value items
            total_events = random.randint(1, 3)
            unique_products = random.randint(1, 2)
            session_duration = random.randint(5, 15)
        
        # Weekend vs weekday behavior
        if is_weekend:
            total_events = int(total_events * 1.2)  # More browsing on weekends
            session_duration = int(session_duration * 1.3)
        
        # Time-based behavior
        if 9 <= hour <= 17:  # Business hours
            session_duration = int(session_duration * 0.8)  # Shorter sessions
        elif 18 <= hour <= 22:  # Evening
            session_duration = int(session_duration * 1.2)  # Longer evening sessions
        
        # Calculate derived features
        avg_price = price
        max_price = price
        total_price_viewed = price * unique_products
        
        # User behavior features (simulated based on session)
        avg_events_per_session = total_events
        total_events_y = total_events
        avg_products_per_session = unique_products
        total_products_viewed = unique_products
        avg_session_duration = session_duration
        
        # Purchase behavior (simulated)
        if price > 200 and is_weekend:
            total_purchases = random.randint(1, 2)  # More likely to buy expensive items on weekends
        elif price > 100:
            total_purchases = random.randint(0, 1)
        else:
            total_purchases = random.randint(0, 1)
        
        total_sessions = random.randint(1, 3)
        avg_price_viewed = total_price_viewed / max(1, total_products_viewed)
        
        # Conversion rate
        conversion_rate = total_purchases / max(1, total_sessions)
        
        # Add all features
        user_data['total_events_x'] = total_events
        user_data['unique_products'] = unique_products
        user_data['unique_categories'] = random.randint(1, min(3, unique_products))
        user_data['avg_price'] = avg_price
        user_data['max_price'] = max_price
        user_data['total_price_viewed'] = total_price_viewed
        user_data['session_duration'] = session_duration
        user_data['avg_events_per_session'] = avg_events_per_session
        user_data['total_events_y'] = total_events_y
        user_data['avg_products_per_session'] = avg_products_per_session
        user_data['total_products_viewed'] = total_products_viewed
        user_data['avg_session_duration'] = avg_session_duration
        user_data['total_purchases'] = total_purchases
        user_data['total_sessions'] = total_sessions
        user_data['avg_price_viewed'] = avg_price_viewed
        user_data['conversion_rate'] = conversion_rate
        user_data['purchased'] = 0  # This is what we're predicting
        
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
        print(f"Error generating features: {e}")
        return None

@app.post("/predict_purchase")
async def predict_purchase(user_data: UserData):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([user_data.model_dump()])
        
        # Generate realistic features
        processed_data = generate_realistic_features(df)
        
        if processed_data is None:
            raise HTTPException(status_code=500, detail="Failed to process user data")
        
        # Make prediction
        probability = model.predict_proba(processed_data)[:, 1][0]
        
        # Generate business insights
        insights = generate_business_insights(probability, user_data)
        
        return {
            "user_id": user_data.user_id,
            "purchase_probability": float(probability),
            "recommendation": "high" if probability > 0.7 else "medium" if probability > 0.5 else "low",
            "business_insights": insights,
            "generated_features": {
                "session_duration_minutes": int(processed_data['session_duration'].iloc[0]),
                "total_products_viewed": int(processed_data['total_products_viewed'].iloc[0]),
                "conversion_rate": float(processed_data['conversion_rate'].iloc[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_business_insights(probability, user_data):
    """Generate business insights based on prediction"""
    insights = []
    
    if probability > 0.7:
        insights.extend([
            "🎯 HIGH CONVERSION POTENTIAL - Show aggressive promotions",
            "💰 Consider premium placement and featured deals",
            "📧 Send personalized email campaigns",
            "🎁 Offer limited-time discounts"
        ])
    elif probability > 0.5:
        insights.extend([
            "📈 MEDIUM CONVERSION POTENTIAL - Show standard recommendations",
            "🔄 Cross-sell related products",
            "📱 Optimize mobile experience",
            "⭐ Show customer reviews and ratings"
        ])
    else:
        insights.extend([
            "🔍 LOW CONVERSION POTENTIAL - Focus on discovery",
            "📚 Show educational content and guides",
            "🎨 Highlight product benefits and features",
            "🤝 Build trust through testimonials"
        ])
    
    # Add time-based insights
    if user_data.is_weekend:
        insights.append("📅 Weekend shopper - Consider leisure-focused messaging")
    
    if 18 <= user_data.hour <= 22:
        insights.append("🌙 Evening shopper - May have more time to browse")
    
    # Add price-based insights
    if user_data.price > 500:
        insights.append("💎 High-value item - Focus on quality and warranty")
    elif user_data.price < 50:
        insights.append("🛍️ Low-value item - Consider bulk offers and add-ons")
    
    return insights

@app.get("/")
async def root():
    return {
        "message": "Enhanced E-Commerce Purchase Prediction API",
        "status": "running",
        "model_info": deployment_package['model_metadata'],
        "endpoints": {
            "predict": "/predict_purchase",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced E-Commerce ML API...")
    print(f"📊 Model: {deployment_package['model_metadata']['algorithm']}")
    print(f"📈 Performance: F1={deployment_package['model_metadata']['f1_score']:.4f}, AUC={deployment_package['model_metadata']['roc_auc']:.4f}")
    uvicorn.run(app, host="0.0.0.0", port=8000)


