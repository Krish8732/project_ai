#!/usr/bin/env python3
"""
E-Commerce Behavior Dataset: Complete Machine Learning Training Guide
Step-by-step implementation following the guide
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os

def step1_data_loading_and_exploration():
    """
    Step 1: Data Loading and Initial Exploration
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING AND INITIAL EXPLORATION")
    print("=" * 60)
    
    try:
        # Load dataset (start with smaller file for testing)
        print("Loading dataset: 2019-Oct.csv...")
        df = pd.read_csv('2019-Oct.csv')
        
        # Basic exploration
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nEvent types:\n{df['event_type'].value_counts()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Display first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Basic statistics
        print(f"\nBasic statistics:")
        print(df.describe())
        
        return df
        
    except FileNotFoundError:
        print("Error: 2019-Oct.csv file not found!")
        print("Please ensure the CSV files are in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def step2_data_preprocessing_and_cleaning(df):
    """
    Step 2: Data Preprocessing and Cleaning
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING AND CLEANING")
    print("=" * 60)
    
    if df is None:
        print("Error: No data to preprocess!")
        return None
    
    print("Original dataset shape:", df.shape)
    
    # 1. Handle missing values
    print("\n1. Handling missing values...")
    df['brand'] = df['brand'].fillna('unknown')
    df['category_code'] = df['category_code'].fillna('other')
    
    print(f"Missing values after filling:")
    print(df.isnull().sum())
    
    # 2. Parse datetime
    print("\n2. Parsing datetime...")
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print(f"Temporal features created: hour, day_of_week, is_weekend")
    
    # 3. Remove invalid data
    print("\n3. Removing invalid data...")
    initial_rows = len(df)
    
    # Remove zero/negative prices
    df = df[df['price'] > 0]
    print(f"Removed {initial_rows - len(df)} rows with invalid prices")
    
    # Remove rows with missing essential IDs
    df = df.dropna(subset=['user_id', 'product_id'])
    print(f"Removed {initial_rows - len(df)} rows with missing essential IDs")
    
    # 4. Create target variable (purchased = 1, others = 0)
    print("\n4. Creating target variable...")
    df['purchased'] = (df['event_type'] == 'purchase').astype(int)
    
    print(f"Target variable 'purchased' created:")
    print(df['purchased'].value_counts())
    
    # 5. Display final dataset info
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    
    return df

def step3_feature_engineering(df):
    """
    Step 3: Feature Engineering (Memory Optimized)
    """
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING (MEMORY OPTIMIZED)")
    print("=" * 60)
    
    if df is None:
        print("Error: No data to engineer features from!")
        return None, None, None
    
    print("Starting feature engineering with memory optimization...")
    
    # Use a smaller sample to avoid memory issues
    sample_size = min(500000, len(df))  # Use 500K records instead of full dataset
    print(f"Using sample of {sample_size:,} records for feature engineering...")
    
    # Sample the data
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    df_sample = df.iloc[sample_indices].copy()
    
    print(f"Sample dataset shape: {df_sample.shape}")
    
    # Session-Level Features
    print("\n1. Creating session-level features...")
    session_features = df_sample.groupby(['user_id', 'user_session']).agg({
        'event_type': 'count',  # Total events in session
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
    
    # Calculate session duration in minutes
    session_features['session_duration'] = (
        session_features['session_end'] - session_features['session_start']
    ).dt.total_seconds() / 60
    
    print(f"Session features created: {session_features.shape}")
    
    # User-Level Features
    print("\n2. Creating user-level features...")
    user_features = session_features.groupby('user_id').agg({
        'total_events': ['mean', 'sum'],
        'unique_products': ['mean', 'sum'],
        'session_duration': 'mean',
        'purchased': ['sum', 'count'],  # Total purchases and sessions
        'avg_price': 'mean'
    }).reset_index()
    
    # Flatten column names
    user_features.columns = ['user_id', 'avg_events_per_session', 'total_events',
                            'avg_products_per_session', 'total_products_viewed',
                            'avg_session_duration', 'total_purchases', 'total_sessions',
                            'avg_price_viewed']
    
    # Calculate conversion rate per user
    user_features['conversion_rate'] = (
        user_features['total_purchases'] / user_features['total_sessions']
    )
    
    print(f"User features created: {user_features.shape}")
    
    # Merge features back to sample dataset (much smaller now)
    print("\n3. Merging features back to sample dataset...")
    
    # Merge session features
    df_with_session = df_sample.merge(
        session_features[['user_id', 'user_session', 'total_events', 'unique_products', 
                         'unique_categories', 'avg_price', 'max_price', 'total_price_viewed',
                         'session_duration']], 
        on=['user_id', 'user_session'], 
        how='left'
    )
    
    # Merge user features
    df_with_features = df_with_session.merge(
        user_features, 
        on='user_id', 
        how='left'
    )
    
    print(f"Final sample dataset with features: {df_with_features.shape}")
    print(f"Final columns: {list(df_with_features.columns)}")
    
    return df_with_features, session_features, user_features

def sample_csv_rows(filename: str, target_rows: int, chunksize: int = 1000000, random_state: int = 42):
    """
    Memory-aware approximate sampler: reads CSV in chunks and samples a fraction
    from each chunk to reach approximately target_rows.
    """
    print(f"Sampling ~{target_rows:,} rows from {filename}...")
    rng_state = random_state
    samples = []
    rows_collected = 0
    try:
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            if rows_collected >= target_rows:
                break
            remaining = max(1, target_rows - rows_collected)
            # Fraction based on remaining vs chunk size, capped at 1
            frac = min(1.0, remaining / max(1, len(chunk)))
            chunk_sample = chunk.sample(frac=frac, random_state=rng_state)
            samples.append(chunk_sample)
            rows_collected += len(chunk_sample)
    except FileNotFoundError:
        print(f"Warning: file not found: {filename}. Skipping.")
    except Exception as e:
        print(f"Error sampling {filename}: {e}")
    if not samples:
        return pd.DataFrame()
    df_sampled = pd.concat(samples, ignore_index=True)
    print(f"Collected {len(df_sampled):,} rows from {filename}")
    return df_sampled

def load_multi_month_sample(file_list, rows_per_month: int = 200000):
    """
    Load and sample multiple monthly CSVs to build a multi-month dataset.
    """
    print("\n" + "=" * 60)
    print("MULTI-MONTH DATA LOADING (SAMPLED PER MONTH)")
    print("=" * 60)
    samples = []
    for fname in file_list:
        if not os.path.exists(fname):
            print(f"File not present, skipping: {fname}")
            continue
        month_df = sample_csv_rows(fname, target_rows=rows_per_month)
        if not month_df.empty:
            samples.append(month_df)
    if not samples:
        print("No files loaded. Aborting multi-month load.")
        return None
    df_all = pd.concat(samples, ignore_index=True)
    print(f"Combined multi-month sample shape: {df_all.shape}")
    return df_all

def step4_time_split_and_imbalance(df_with_features):
    """
    Time-based split: Train Oct 2019 - Feb 2020, Val Mar 2020, Test Apr 2020.
    Apply SMOTE only on training data. Label-encode categorical features using train fit.
    """
    print("\n" + "=" * 60)
    print("STEP 4B: TIME-BASED SPLIT AND IMBALANCE HANDLING")
    print("=" * 60)

    if df_with_features is None or df_with_features.empty:
        print("Error: No data provided for time-based split.")
        return None

    # Ensure event_time is datetime (handle timezone-aware datetimes)
    if not pd.api.types.is_datetime64_any_dtype(df_with_features['event_time']):
        df_with_features['event_time'] = pd.to_datetime(df_with_features['event_time'])
    # Convert timezone-aware to naive datetime if needed
    if df_with_features['event_time'].dt.tz is not None:
        df_with_features['event_time'] = df_with_features['event_time'].dt.tz_localize(None)

    # Define masks by month
    train_mask = (df_with_features['event_time'] >= '2019-10-01') & (df_with_features['event_time'] < '2020-03-01')
    val_mask = (df_with_features['event_time'] >= '2020-03-01') & (df_with_features['event_time'] < '2020-04-01')
    test_mask = (df_with_features['event_time'] >= '2020-04-01') & (df_with_features['event_time'] < '2020-05-01')

    print(f"Train rows: {train_mask.sum():,}, Val rows: {val_mask.sum():,}, Test rows: {test_mask.sum():,}")

    numerical_features = ['price', 'hour', 'day_of_week', 'is_weekend',
                          'total_events_x', 'unique_products', 'unique_categories',
                          'avg_price', 'max_price', 'total_price_viewed', 'session_duration',
                          'avg_events_per_session', 'total_events_y', 'avg_products_per_session',
                          'total_products_viewed', 'avg_session_duration', 'total_purchases',
                          'total_sessions', 'avg_price_viewed', 'conversion_rate']
    categorical_features = ['brand', 'category_code']

    feature_cols = numerical_features + categorical_features

    df_with_features[numerical_features] = df_with_features[numerical_features].fillna(0)
    df_with_features[categorical_features] = df_with_features[categorical_features].fillna('unknown')

    X_train = df_with_features.loc[train_mask, feature_cols].copy()
    y_train = df_with_features.loc[train_mask, 'purchased'].astype(int)
    X_val = df_with_features.loc[val_mask, feature_cols].copy()
    y_val = df_with_features.loc[val_mask, 'purchased'].astype(int)
    X_test = df_with_features.loc[test_mask, feature_cols].copy()
    y_test = df_with_features.loc[test_mask, 'purchased'].astype(int)

    print(f"Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Label encode categorical using train fit
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Handle unseen labels in val/test by mapping unknown
        def transform_safe(series):
            known = set(le.classes_)
            mapped = series.astype(str).where(series.astype(str).isin(known), other='__unknown__')
            if '__unknown__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__unknown__')
            return le.transform(mapped)
        X_val[col] = transform_safe(X_val[col])
        X_test[col] = transform_safe(X_test[col])
        le_dict[col] = le
        print(f"Encoded {col} (train classes: {len(le.classes_)})")

    # SMOTE on training only
    print("Applying SMOTE on training split only...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Train before SMOTE: {np.bincount(y_train) if len(y_train)>0 else []}")
    print(f"Train after  SMOTE: {np.bincount(y_train_res) if len(y_train_res)>0 else []}")

    return X_train_res, X_val, X_test, y_train_res, y_val, y_test, le_dict

def step5_model_training_and_validation(X_train, X_val, X_test, y_train, y_val, y_test, le_dict):
    """
    Step 5: Model Training and Validation
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING AND VALIDATION")
    print("=" * 60)
    
    if X_train is None or y_train is None:
        print("Error: No training data available!")
        return None, None, None
    
    print("Starting model training...")
    
    # 1. Train CatBoost Model (Recommended)
    print("\n1. Training CatBoost Model...")
    
    try:
        from catboost import CatBoostClassifier
        
        # Define categorical features for CatBoost
        categorical_features = ['brand', 'category_code']
        cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_features]
        
        # Initialize CatBoost
        catboost_model = CatBoostClassifier(
            iterations=500,  # Reduced for faster training
            learning_rate=0.1,
            depth=6,
            class_weights=[1, 3],  # Handle class imbalance
            verbose=100,
            random_seed=42,
            task_type='CPU'  # Use CPU for compatibility
        )
        
        print("Training CatBoost model...")
        catboost_model.fit(
            X_train, y_train,
            cat_features=cat_features_indices,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        print("✅ CatBoost training completed!")
        
    except Exception as e:
        print(f"Error training CatBoost: {e}")
        catboost_model = None
    
    # 2. Train XGBoost Model
    print("\n2. Training XGBoost Model...")
    
    try:
        import xgboost as xgb
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 3,  # Handle class imbalance
            'random_state': 42,
            'n_jobs': -1
        }
        
        print("Training XGBoost model...")
        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,  # Reduced for faster training
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        print("✅ XGBoost training completed!")
        
    except Exception as e:
        print(f"Error training XGBoost: {e}")
        xgb_model = None
    
    # 3. Train Random Forest (Baseline)
    print("\n3. Training Random Forest Model...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        print("Training Random Forest model...")
        rf_model.fit(X_train, y_train)
        
        print("✅ Random Forest training completed!")
        
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        rf_model = None
    
    # 4. Model Performance Summary
    print("\n4. Model Training Summary:")
    models = {}
    
    if catboost_model is not None:
        models['CatBoost'] = catboost_model
        print("✅ CatBoost: Trained successfully")
    
    if xgb_model is not None:
        models['XGBoost'] = xgb_model
        print("✅ XGBoost: Trained successfully")
    
    if rf_model is not None:
        models['RandomForest'] = rf_model
        print("✅ Random Forest: Trained successfully")
    
    if not models:
        print("❌ No models were trained successfully!")
        return None, None, None
    
    print(f"\nTotal models trained: {len(models)}")
    
    return models, X_train, X_val, X_test, y_train, y_val, y_test

def step6_model_evaluation(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Step 6: Model Evaluation
    """
    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 60)
    
    if not models:
        print("Error: No models to evaluate!")
        return None
    
    # Local import to ensure availability when evaluating XGBoost
    try:
        import xgboost as xgb  # noqa: F401
    except Exception:
        xgb = None
    
    print("Starting comprehensive model evaluation...")
    
    # 1. Model Performance Comparison
    print("\n1. Model Performance Comparison:")
    print("-" * 80)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Get predictions
            if model_name == 'XGBoost' and xgb is not None:
                # XGBoost needs DMatrix format
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = model.predict(dtest)
            else:
                # Other models use predict_proba
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Business metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results[model_name] = {
                'auc': auc_score,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
            
            print(f"✅ {model_name} Evaluation:")
            print(f"   ROC AUC: {auc_score:.4f}")
            print(f"   F1 Score: {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   True Positives: {tp}")
            print(f"   False Positives: {fp}")
            print(f"   False Negatives: {fn}")
            print(f"   True Negatives: {tn}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            results[model_name] = None
    
    # 2. Model Ranking
    print("\n2. Model Ranking by Performance:")
    print("-" * 80)
    
    # Filter out failed models
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Sort by F1 score (most important for imbalanced data)
        sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        print("🏆 Model Rankings (by F1 Score):")
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            print(f"{i}. {model_name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        
        # Best model
        best_model_name = sorted_models[0][0]
        best_model = models[best_model_name]
        print(f"\n🥇 Best Model: {best_model_name}")
        
    else:
        print("❌ No models evaluated successfully!")
        return None
    
    # 3. Business Impact Analysis
    print("\n3. Business Impact Analysis:")
    print("-" * 80)
    
    if best_model_name in valid_results:
        metrics = valid_results[best_model_name]
        
        print(f"Using {best_model_name} for business analysis:")
        print(f"📊 Conversion Capture Rate (Recall): {metrics['recall']:.2%}")
        print(f"🎯 Precision (Quality of Predictions): {metrics['precision']:.2%}")
        print(f"💰 False Positive Cost: {metrics['fp']} incorrect purchase predictions")
        print(f"💸 False Negative Cost: {metrics['fn']} missed purchase opportunities")
        
        # Calculate business metrics at different thresholds
        print(f"\n📈 Performance at Different Thresholds:")
        
        try:
            if best_model_name == 'XGBoost' and xgb is not None:
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = best_model.predict(dtest)
            else:
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_pred_proba = None
        
        if y_pred_proba is not None:
            thresholds = [0.3, 0.5, 0.7]
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba > threshold).astype(int)
                tp_thresh = np.sum((y_test == 1) & (y_pred_thresh == 1))
                fp_thresh = np.sum((y_test == 0) & (y_pred_thresh == 1))
                fn_thresh = np.sum((y_test == 1) & (y_pred_thresh == 0))
                
                precision_thresh = tp_thresh / (tp_thresh + fp_thresh) if (tp_thresh + fp_thresh) > 0 else 0
                recall_thresh = tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0
                
                print(f"   Threshold {threshold}: Precision={precision_thresh:.3f}, Recall={recall_thresh:.3f}")
    
    # 4. Final Summary
    print("\n4. Final Project Summary:")
    print("-" * 80)
    print("🎉 E-COMMERCE ML PROJECT COMPLETED SUCCESSFULLY!")
    print(f"📊 Dataset: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,} total samples")
    print(f"🔧 Features: {X_train.shape[1]} engineered features")
    print(f"🤖 Models: {len(models)} trained and evaluated")
    print(f"🏆 Best Model: {best_model_name}")
    
    if best_model_name in valid_results:
        best_metrics = valid_results[best_model_name]
        print(f"📈 Best Performance: F1={best_metrics['f1']:.4f}, AUC={best_metrics['auc']:.4f}")
    
    print("\n🚀 Next Steps:")
    print("   - Deploy best model to production")
    print("   - Monitor performance over time")
    print("   - Retrain with new data periodically")
    print("   - A/B test model improvements")
    
    return best_model_name, valid_results

def run_multi_month_pipeline():
    """
    Orchestrate multi-month training using time-based split.
    """
    # Define file list in chronological order
    files = [
        '2019-Oct.csv',
        '2019-Nov.csv',
        '2019-Dec.csv',
        '2020-Jan.csv',
        '2020-Feb.csv',
        '2020-Mar.csv',
        '2020-Apr.csv',
    ]
    df_multi = load_multi_month_sample(files, rows_per_month=150000)
    if df_multi is None:
        return

    # Reuse Step 2 preprocessing
    df_clean = step2_data_preprocessing_and_cleaning(df_multi)
    if df_clean is None:
        return

    # Reuse Step 3 feature engineering (will internally sample to 500k)
    df_features, session_features, user_features = step3_feature_engineering(df_clean)
    if df_features is None:
        return

    # Time-based split + SMOTE on train
    split = step4_time_split_and_imbalance(df_features)
    if split is None:
        return
    X_train, X_val, X_test, y_train, y_val, y_test, le_dict = split

    # Train models
    models_result = step5_model_training_and_validation(X_train, X_val, X_test, y_train, y_val, y_test, le_dict)
    if models_result is None:
        return
    models, X_train, X_val, X_test, y_train, y_val, y_test = models_result

    # Evaluate
    step6_model_evaluation(models, X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    print("E-Commerce ML Training Project")
    print("Starting with Step 1: Data Loading and Exploration")
    
    # Execute Step 1
    df = step1_data_loading_and_exploration()
    
    if df is not None:
        print(f"\n✅ Step 1 completed successfully!")
        print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Execute Step 2
        df_cleaned = step2_data_preprocessing_and_cleaning(df)
        
        if df_cleaned is not None:
            print(f"\n✅ Step 2 completed successfully!")
            print(f"Cleaned dataset has {len(df_cleaned)} rows and {len(df_cleaned.columns)} columns")
            
            # Execute Step 3
            df_with_features, session_features, user_features = step3_feature_engineering(df_cleaned)
            
            if df_with_features is not None:
                print(f"\n✅ Step 3 completed successfully!")
                print(f"Dataset with features has {len(df_with_features)} rows and {len(df_with_features.columns)} columns")
                
                # Execute Step 4
                result = step4_time_split_and_imbalance(df_with_features)
                
                if result is not None:
                    X_train, X_val, X_test, y_train, y_val, y_test, le_dict = result
                    print(f"\n✅ Step 4 completed successfully!")
                    print(f"Data split ready for model training!")
                    
                    # Execute Step 5
                    models_result = step5_model_training_and_validation(X_train, X_val, X_test, y_train, y_val, y_test, le_dict)
                    
                    if models_result is not None:
                        models, X_train, X_val, X_test, y_train, y_val, y_test = models_result
                        print(f"\n✅ Step 5 completed successfully!")
                        print(f"Models trained: {list(models.keys())}")
                        
                        # Execute Step 6
                        evaluation_result = step6_model_evaluation(models, X_train, X_val, X_test, y_train, y_val, y_test)
                        
                        if evaluation_result is not None:
                            best_model_name, valid_results = evaluation_result
                            print(f"\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
                            print(f"Best performing model: {best_model_name}")
                        else:
                            print("\n❌ Step 6 failed. Please check the error messages above.")
                    else:
                        print("\n❌ Step 5 failed. Please check the error messages above.")
                else:
                    print("\n❌ Step 4 failed. Please check the error messages above.")
            else:
                print("\n❌ Step 3 failed. Please check the error messages above.")
        else:
            print("\n❌ Step 2 failed. Please check the error messages above.")
    else:
        print("\n❌ Step 1 failed. Please check the error messages above.")
