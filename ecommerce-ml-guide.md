# E-Commerce Behavior Dataset: Complete Machine Learning Training Guide

## Dataset Overview

The **eCommerce behavior data from multi-category store** dataset contains behavioral data for **285 million user events** from a large multi-category eCommerce store spanning **7 months (October 2019 - April 2020)**. This dataset is ideal for training machine learning models to predict user purchase behavior and analyze customer patterns.

### Dataset Structure

**Files Available:**
- `2019-Oct.csv.gz` (1.62GB)
- `2019-Nov.csv.gz` (2.69GB)  
- `2019-Dec.csv.gz` (2.74GB)
- `2020-Jan.csv.gz` (2.23GB)
- `2020-Feb.csv.gz` (2.19GB)
- `2020-Mar.csv.gz` (2.25GB)
- `2020-Apr.csv.gz` (2.73GB)

**Columns (9 total):**
- `event_time`: Timestamp of the event in UTC
- `event_type`: Type of event (view, cart, remove_from_cart, purchase)
- `product_id`: Unique identifier for a product
- `category_id`: Unique identifier for product category
- `category_code`: Product's category taxonomy (if available)
- `brand`: Brand name (lowercase, may be missing)
- `price`: Float price of the product
- `user_id`: Unique identifier for a user
- `user_session`: Temporary session ID (changes after inactivity)

## Which Datasets to Use

**Recommendation: Use both datasets strategically**

1. **For Initial Development**: Start with smaller files (Oct-Nov 2019) for faster prototyping
2. **For Production Models**: Use all 7 months of data for better generalization
3. **Time-based Splitting**: 
   - **Training**: Oct 2019 - Feb 2020
   - **Validation**: March 2020
   - **Test**: April 2020

## Best Fit Algorithms

Based on research analysis, here are the top-performing algorithms for this dataset:

### **Classification Tasks (Purchase Prediction)**

#### **1. CatBoost (Recommended)**
- **F1 Score**: 0.93
- **ROC AUC**: 0.985
- **Advantages**: Handles categorical features well, robust to overfitting

#### **2. XGBoost**  
- **F1 Score**: 0.92
- **ROC AUC**: 0.984
- **Advantages**: Excellent performance, handles missing values, fast training

#### **3. Random Forest**
- **F1 Score**: 0.82
- **Accuracy**: 0.96
- **Advantages**: Interpretable, handles mixed data types, good baseline model

#### **4. LightGBM**
- **Advantages**: Fast training, memory efficient, good for large datasets

### **Deep Learning Models**

#### **1. LSTM Networks**
- **Use Case**: Sequential behavior modeling
- **Performance**: 88-96% accuracy
- **Advantages**: Captures temporal patterns in user sessions

#### **2. Neural Networks**
- **Performance**: AUC 0.88-0.97
- **Advantages**: Handles complex non-linear relationships

## Complete Training Steps

### **Step 1: Data Loading and Initial Exploration**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset (start with smaller file for testing)
df = pd.read_csv('2019-Oct.csv.gz', compression='gzip')

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"Event types: {df['event_type'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")
```

### **Step 2: Data Preprocessing and Cleaning**

#### **Essential Preprocessing Steps:**

```python
# 1. Handle missing values
df['brand'] = df['brand'].fillna('unknown')
df['category_code'] = df['category_code'].fillna('other')

# 2. Parse datetime
df['event_time'] = pd.to_datetime(df['event_time'])
df['hour'] = df['event_time'].dt.hour
df['day_of_week'] = df['event_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 3. Remove invalid data
df = df[df['price'] > 0]  # Remove zero/negative prices
df = df.dropna(subset=['user_id', 'product_id'])  # Essential IDs

# 4. Create target variable (purchased = 1, others = 0)
df['purchased'] = (df['event_type'] == 'purchase').astype(int)
```

### **Step 3: Feature Engineering**

#### **Session-Level Features:**

```python
# Group by user session to create session-level features
session_features = df.groupby(['user_id', 'user_session']).agg({
    'event_type': ['count'],  # Total events in session
    'product_id': 'nunique',  # Unique products viewed
    'category_id': 'nunique',  # Unique categories
    'price': ['mean', 'max', 'sum'],  # Price statistics
    'purchased': 'max',  # Whether any purchase occurred
    'event_time': ['min', 'max']  # Session duration
}).reset_index()

# Flatten column names
session_features.columns = ['user_id', 'user_session', 'total_events', 
                           'unique_products', 'unique_categories',
                           'avg_price', 'max_price', 'total_price_viewed',
                           'purchased', 'session_start', 'session_end']

# Calculate session duration
session_features['session_duration'] = (
    session_features['session_end'] - session_features['session_start']
).dt.total_seconds() / 60  # in minutes
```

#### **User-Level Features:**

```python
# Create user behavior features
user_features = session_features.groupby('user_id').agg({
    'total_events': ['mean', 'sum'],
    'unique_products': ['mean', 'sum'],
    'session_duration': 'mean',
    'purchased': ['sum', 'count'],  # Total purchases and sessions
    'avg_price': 'mean'
}).reset_index()

# Calculate conversion rate per user
user_features['conversion_rate'] = (
    user_features[('purchased', 'sum')] / user_features[('purchased', 'count')]
)
```

### **Step 4: Handle Class Imbalance**

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Check class distribution
print(f"Class distribution: {Counter(y)}")

# Apply SMOTE for oversampling minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"After SMOTE: {Counter(y_resampled)}")
```

### **Step 5: Model Training and Validation**

#### **CatBoost Model (Recommended):**

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

# Define categorical features
categorical_features = ['brand', 'category_code', 'hour', 'day_of_week']

# Initialize CatBoost
catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    class_weights=[1, 3],  # Handle class imbalance
    verbose=100,
    random_seed=42
)

# Train model
catboost_model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)
```

#### **XGBoost Model:**

```python
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_val[col] = le.transform(X_val[col].astype(str))
    le_dict[col] = le

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 3,  # Handle class imbalance
    'random_state': 42
}

# Train XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=100
)
```

### **Step 6: Model Evaluation**

#### **Critical Evaluation Metrics for Imbalanced Data:**

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, f1_score

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Essential metrics for imbalanced datasets
print("=== MODEL EVALUATION ===")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
```

#### **Additional Evaluation for E-commerce:**

```python
# Business metrics
def calculate_business_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # True/False Positives and Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Business metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'true_positives': tp,
        'false_positives': fp,
        'conversion_capture_rate': recall
    }

# Evaluate at different thresholds
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    metrics = calculate_business_metrics(y_test, y_pred_proba, threshold)
    print(f"Threshold {threshold}: {metrics}")
```

## Must-Do Steps for Success

### **1. Data Quality Checks**
- ✅ Remove duplicate events
- ✅ Handle missing values appropriately
- ✅ Validate timestamp consistency
- ✅ Remove invalid price entries (≤ 0)

### **2. Feature Engineering Essentials**
- ✅ Create temporal features (hour, day, weekend)
- ✅ Calculate session-level aggregations
- ✅ Build user behavior profiles
- ✅ Engineer conversion funnel features

### **3. Handle Class Imbalance**
- ✅ Use appropriate sampling techniques (SMOTE, undersampling)
- ✅ Apply class weights in models
- ✅ Use proper evaluation metrics (F1, ROC-AUC, not just accuracy)

### **4. Proper Validation Strategy**
- ✅ Time-based splitting (avoid data leakage)
- ✅ Cross-validation with temporal awareness
- ✅ Separate validation and test sets

### **5. Model Selection Criteria**
- ✅ Prioritize F1 score and ROC-AUC over accuracy
- ✅ Consider business impact of false positives vs false negatives
- ✅ Evaluate model interpretability requirements

### **6. Production Considerations**
- ✅ Monitor model performance over time
- ✅ Implement proper logging and tracking
- ✅ Plan for model retraining schedule
- ✅ Consider computational efficiency for real-time predictions

## Advanced Tips

1. **Ensemble Methods**: Combine CatBoost + XGBoost for better performance
2. **Sequential Modeling**: Use LSTM for capturing session sequences
3. **Feature Selection**: Use feature importance from tree-based models
4. **Hyperparameter Tuning**: Use Optuna or GridSearch for optimization
5. **A/B Testing**: Validate model improvements with business metrics

This comprehensive guide provides a solid foundation for training high-performance machine learning models on the e-commerce behavior dataset. Start with the recommended CatBoost or XGBoost models for best results!