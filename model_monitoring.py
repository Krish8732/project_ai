
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
