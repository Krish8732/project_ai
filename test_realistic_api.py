#!/usr/bin/env python3
"""
Test the API with realistic feature values that match the training data
"""

import requests
import json
import time

def test_api_with_realistic_data():
    """Test the API with realistic feature values"""
    print("🧪 TESTING API WITH REALISTIC DATA")
    print("=" * 50)
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Test case 1: High-value electronics (likely to purchase)
    print("\n📱 Test Case 1: High-value Electronics")
    print("-" * 40)
    
    high_value_data = {
        "user_id": "user_high_value",
        "session_id": "session_001",
        "product_id": "iphone_15",
        "category_code": "electronics.smartphones",
        "brand": "apple",
        "price": 999.99,
        "hour": 15,
        "day_of_week": 3,
        "is_weekend": 0
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_purchase",
            json=high_value_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            prob = result['purchase_probability']
            if prob > 0.7:
                print(f"🎯 HIGH probability - Show aggressive promotions!")
            elif prob > 0.5:
                print(f"📈 MEDIUM probability - Show standard recommendations")
            else:
                print(f"🔍 LOW probability - Show discovery content")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test case 2: Weekend sports shopping
    print("\n🏃 Test Case 2: Weekend Sports Shopping")
    print("-" * 40)
    
    weekend_sports_data = {
        "user_id": "user_weekend",
        "session_id": "session_002",
        "product_id": "nike_shoes",
        "category_code": "sports.shoes",
        "brand": "nike",
        "price": 89.99,
        "hour": 14,
        "day_of_week": 6,
        "is_weekend": 1
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_purchase",
            json=weekend_sports_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            prob = result['purchase_probability']
            if prob > 0.7:
                print(f"🎯 HIGH probability - Show aggressive promotions!")
            elif prob > 0.5:
                print(f"📈 MEDIUM probability - Show standard recommendations")
            else:
                print(f"🔍 LOW probability - Show discovery content")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test case 3: Low-value home items
    print("\n🏠 Test Case 3: Low-value Home Items")
    print("-" * 40)
    
    low_value_data = {
        "user_id": "user_low_value",
        "session_id": "session_003",
        "product_id": "kitchen_spoon",
        "category_code": "home.kitchen",
        "brand": "generic",
        "price": 19.99,
        "hour": 10,
        "day_of_week": 1,
        "is_weekend": 0
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_purchase",
            json=low_value_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            prob = result['purchase_probability']
            if prob > 0.7:
                print(f"🎯 HIGH probability - Show aggressive promotions!")
            elif prob > 0.5:
                print(f"📈 MEDIUM probability - Show standard recommendations")
            else:
                print(f"🔍 LOW probability - Show discovery content")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and accessible")
            return True
        else:
            print(f"⚠️ API responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🚀 REALISTIC API TESTING")
    print("=" * 50)
    
    # Check API health first
    if test_api_health():
        test_api_with_realistic_data()
    else:
        print("Please start the API first with: python api_example.py")


