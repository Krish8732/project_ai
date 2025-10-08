import requests
import json

URL = "http://127.0.0.1:8001/predict_event"

sample_features = {
    'brand': 'apple',
    'category_code': 'electronics.smartphones',
    'price': 999.99,
    'hour': 15,
    'day_of_week': 2,
    'is_weekend': 0
}

def main():
    try:
        resp = requests.post(URL, json=sample_features, timeout=10)
        print('Status:', resp.status_code)
        print('Response:', json.dumps(resp.json(), indent=2))
    except Exception as e:
        print('Request failed:', e)

if __name__ == '__main__':
    main()
