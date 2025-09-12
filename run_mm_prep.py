import joblib
import ecommerce_ml_training as m

if __name__ == "__main__":
    files = [
        '2019-Oct.csv',
        '2019-Nov.csv',
        '2019-Dec.csv',
        '2020-Jan.csv',
        '2020-Feb.csv',
        '2020-Mar.csv',
        '2020-Apr.csv',
    ]

    df_multi = m.load_multi_month_sample(files, rows_per_month=150000)
    if df_multi is None:
        print("SPLIT_FAILED:LOAD")
        raise SystemExit(1)

    df_clean = m.step2_data_preprocessing_and_cleaning(df_multi)
    if df_clean is None:
        print("SPLIT_FAILED:CLEAN")
        raise SystemExit(1)

    df_feat, _, _ = m.step3_feature_engineering(df_clean)
    if df_feat is None:
        print("SPLIT_FAILED:FEATURES")
        raise SystemExit(1)

    split = m.step4_time_split_and_imbalance(df_feat)
    if split is None:
        print("SPLIT_FAILED:SPLIT")
        raise SystemExit(1)

    X_train, X_val, X_test, y_train, y_val, y_test, le_dict = split
    joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test, le_dict), 'mm_split.pkl')
    print("SPLIT_READY", X_train.shape, X_val.shape, X_test.shape)



