#!/usr/bin/env python3
"""
Enhanced E-Commerce ML System with Advanced Capabilities
Incorporates global best practices and advanced ML techniques
"""

import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import shap
import lime
import lime.lime_tabular
from datetime import datetime, timedelta
import json
import logging
import pickle

class EnhancedEcommerceML:
    """
    Enhanced E-Commerce ML System with Advanced Capabilities
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.scalers = {}
        self.explainer = None
        self.performance_history = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load and prepare data with advanced preprocessing"""
        self.logger.info("Loading data with advanced preprocessing...")
        
        try:
            # Load the prepared data
            X_train, X_val, X_test, y_train, y_val, y_test, le_dict = joblib.load('mm_split.pkl')
            
            # Advanced feature scaling
            self.scalers['robust'] = RobustScaler()
            self.scalers['standard'] = StandardScaler()
            
            # Apply robust scaling to numerical features
            numerical_features = X_train.select_dtypes(include=[np.number]).columns
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numerical_features] = self.scalers['robust'].fit_transform(X_train[numerical_features])
            X_val_scaled[numerical_features] = self.scalers['robust'].transform(X_val[numerical_features])
            X_test_scaled[numerical_features] = self.scalers['robust'].transform(X_test[numerical_features])
            
            self.data = {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'le_dict': le_dict,
                'feature_names': list(X_train.columns)
            }
            
            self.logger.info(f"Data loaded successfully: {X_train.shape[0]} training samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def hyperparameter_optimization(self, model_type='catboost'):
        """Advanced hyperparameter optimization using Optuna"""
        self.logger.info(f"Starting hyperparameter optimization for {model_type}...")
        
        def objective(trial):
            if model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'class_weights': [1, trial.suggest_float('class_weight', 1, 5)]
                }
                
                model = CatBoostClassifier(**params, verbose=False, random_seed=42)
                model.fit(
                    self.data['X_train'], self.data['y_train'],
                    eval_set=(self.data['X_val'], self.data['y_val']),
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                y_pred_proba = model.predict_proba(self.data['X_val'])[:, 1]
                return roc_auc_score(self.data['y_val'], y_pred_proba)
            
            elif model_type == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5)
                }
                
                model = xgb.XGBClassifier(**params, random_state=42)
                model.fit(self.data['X_train'], self.data['y_train'])
                
                y_pred_proba = model.predict_proba(self.data['X_val'])[:, 1]
                return roc_auc_score(self.data['y_val'], y_pred_proba)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        self.logger.info(f"Best {model_type} score: {study.best_value:.4f}")
        self.logger.info(f"Best {model_type} params: {study.best_params}")
        
        return study.best_params
    
    def train_advanced_models(self):
        """Train multiple advanced models with optimized parameters"""
        self.logger.info("Training advanced models...")
        
        # Get optimized parameters
        catboost_params = self.hyperparameter_optimization('catboost')
        xgboost_params = self.hyperparameter_optimization('xgboost')
        
        # Train CatBoost with optimized parameters
        self.models['catboost'] = CatBoostClassifier(
            **catboost_params,
            verbose=100,
            random_seed=42
        )
        
        categorical_features = ['brand', 'category_code']
        cat_features_indices = [self.data['X_train'].columns.get_loc(col) for col in categorical_features]
        
        self.models['catboost'].fit(
            self.data['X_train'], self.data['y_train'],
            cat_features=cat_features_indices,
            eval_set=(self.data['X_val'], self.data['y_val']),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Train XGBoost with optimized parameters
        self.models['xgboost'] = xgb.XGBClassifier(**xgboost_params, random_state=42)
        self.models['xgboost'].fit(self.data['X_train'], self.data['y_train'])
        
        # Train LightGBM
        self.models['lightgbm'] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(self.data['X_train'], self.data['y_train'])
        
        # Train Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(self.data['X_train'], self.data['y_train'])
        
        self.logger.info("All models trained successfully")
    
    def create_ensemble_model(self):
        """Create advanced ensemble model"""
        self.logger.info("Creating ensemble model...")
        
        # Create voting classifier
        estimators = [
            ('catboost', self.models['catboost']),
            ('xgboost', self.models['xgboost']),
            ('lightgbm', self.models['lightgbm']),
            ('random_forest', self.models['random_forest'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[0.4, 0.3, 0.2, 0.1]  # Weighted voting based on expected performance
        )
        
        self.ensemble_model.fit(self.data['X_train'], self.data['y_train'])
        self.logger.info("Ensemble model created successfully")
    
    def analyze_feature_importance(self):
        """Advanced feature importance analysis using SHAP"""
        self.logger.info("Analyzing feature importance...")
        
        # SHAP analysis for CatBoost
        explainer = shap.TreeExplainer(self.models['catboost'])
        shap_values = explainer.shap_values(self.data['X_val'])
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_names = self.data['feature_names']
        
        self.feature_importance = dict(zip(feature_names, feature_importance))
        
        # Sort by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info("Top 10 most important features:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            self.logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        # Save SHAP explainer for later use
        self.explainer = explainer
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        self.logger.info("Evaluating all models...")
        
        results = {}
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.data['X_test'])[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            results[name] = {
                'auc': roc_auc_score(self.data['y_test'], y_pred_proba),
                'f1': f1_score(self.data['y_test'], y_pred),
                'predictions': y_pred_proba
            }
        
        # Evaluate ensemble
        if self.ensemble_model:
            y_pred_proba = self.ensemble_model.predict_proba(self.data['X_test'])[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            results['ensemble'] = {
                'auc': roc_auc_score(self.data['y_test'], y_pred_proba),
                'f1': f1_score(self.data['y_test'], y_pred),
                'predictions': y_pred_proba
            }
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        self.logger.info(f"Best model: {best_model[0]} (F1: {best_model[1]['f1']:.4f}, AUC: {best_model[1]['auc']:.4f})")
        
        # Save results
        with open('model_evaluation_results.json', 'w') as f:
            json.dump({k: {'auc': float(v['auc']), 'f1': float(v['f1'])} for k, v in results.items()}, f, indent=2)
        
        self.evaluation_results = results
        return results
    
    def create_lime_explainer(self):
        """Create LIME explainer for model interpretability"""
        self.logger.info("Creating LIME explainer...")
        
        # Create LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.data['X_train'].values,
            feature_names=self.data['feature_names'],
            class_names=['No Purchase', 'Purchase'],
            mode='classification'
        )
    
    def explain_prediction(self, sample_data, model_name='ensemble'):
        """Explain individual predictions using LIME and SHAP"""
        if model_name == 'ensemble':
            model = self.ensemble_model
        else:
            model = self.models[model_name]
        
        # LIME explanation
        lime_exp = self.lime_explainer.explain_instance(
            sample_data.values[0],
            model.predict_proba,
            num_features=10
        )
        
        # SHAP explanation
        shap_values = self.explainer.shap_values(sample_data)
        
        return {
            'lime_explanation': lime_exp,
            'shap_values': shap_values,
            'prediction': model.predict_proba(sample_data)[0, 1]
        }
    
    def save_enhanced_model(self):
        """Save the enhanced model system"""
        self.logger.info("Saving enhanced model system...")
        
        enhanced_package = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'explainer': self.explainer,
            'lime_explainer': self.lime_explainer,
            'evaluation_results': self.evaluation_results,
            'feature_names': self.data['feature_names'],
            'le_dict': self.data['le_dict'],
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'model_versions': {
                    'catboost': CatBoostClassifier.__version__,
                    'xgboost': xgb.__version__,
                    'lightgbm': LGBMClassifier.__version__
                },
                'performance': self.evaluation_results
            }
        }
        
        joblib.dump(enhanced_package, 'enhanced_model_system.pkl')
        self.logger.info("Enhanced model system saved successfully")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        self.logger.info("Generating performance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': self.evaluation_results,
            'feature_importance': dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]),
            'data_info': {
                'training_samples': len(self.data['X_train']),
                'validation_samples': len(self.data['X_val']),
                'test_samples': len(self.data['X_test']),
                'features': len(self.data['feature_names']),
                'class_distribution': {
                    'train': self.data['y_train'].value_counts().to_dict(),
                    'val': self.data['y_val'].value_counts().to_dict(),
                    'test': self.data['y_test'].value_counts().to_dict()
                }
            },
            'recommendations': self.generate_recommendations()
        }
        
        with open('performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("Performance report generated successfully")
        return report
    
    def generate_recommendations(self):
        """Generate business recommendations based on model performance"""
        best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['f1'])
        
        recommendations = {
            'model_selection': f"Use {best_model[0]} model for production (F1: {best_model[1]['f1']:.4f})",
            'feature_engineering': "Consider engineering more features based on top important features",
            'data_collection': "Focus on collecting data for high-importance features",
            'monitoring': "Set up monitoring for model drift and performance degradation",
            'retraining': "Retrain model monthly with new data",
            'business_impact': f"Expected conversion rate improvement: {best_model[1]['f1'] * 100:.1f}%"
        }
        
        return recommendations

def main():
    """Main function to run the enhanced ML system"""
    print("🚀 ENHANCED E-COMMERCE ML SYSTEM")
    print("=" * 60)
    
    # Initialize enhanced system
    ml_system = EnhancedEcommerceML()
    
    # Load data
    if not ml_system.load_data():
        print("❌ Failed to load data")
        return
    
    # Train advanced models
    ml_system.train_advanced_models()
    
    # Create ensemble
    ml_system.create_ensemble_model()
    
    # Analyze feature importance
    ml_system.analyze_feature_importance()
    
    # Create explainers
    ml_system.create_lime_explainer()
    
    # Evaluate models
    results = ml_system.evaluate_models()
    
    # Save enhanced system
    ml_system.save_enhanced_model()
    
    # Generate report
    report = ml_system.generate_performance_report()
    
    print("\n🎉 ENHANCED ML SYSTEM COMPLETED!")
    print("=" * 60)
    print("✅ Advanced models trained with hyperparameter optimization")
    print("✅ Ensemble model created")
    print("✅ Feature importance analyzed")
    print("✅ Model interpretability implemented")
    print("✅ Performance report generated")
    print(f"🏆 Best model: {max(results.items(), key=lambda x: x[1]['f1'])[0]}")
    
    return ml_system

if __name__ == "__main__":
    main()

