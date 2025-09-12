import joblib
import ecommerce_ml_training as m

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, le_dict = joblib.load('mm_split.pkl')
    models_result = m.step5_model_training_and_validation(X_train, X_val, X_test, y_train, y_val, y_test, le_dict)
    if models_result is None:
        raise SystemExit(1)
    models, X_train, X_val, X_test, y_train, y_val, y_test = models_result
    m.step6_model_evaluation(models, X_train, X_val, X_test, y_train, y_val, y_test)



