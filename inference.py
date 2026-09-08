
import pandas as pd
import numpy as np
import joblib
import json



# LOAD ARTIFACTS


model = joblib.load(
    'xgboost_model.pkl'
)

preprocessor = joblib.load(
    'preprocessor.pkl'
)


with open(
    'feature_info.json',
    'r',
    encoding='utf-8'
) as f:
    feature_info = json.load(f)


with open(
    'threshold.json',
    'r',
    encoding='utf-8'
) as f:
    threshold_info = json.load(f)


THRESHOLD = threshold_info['threshold']

EXPECTED_FEATURES = (
    feature_info['raw_input_features']
)



# INFERENCE FUNCTION


def detect(input_data):

    if isinstance(input_data, dict):

        input_data = pd.DataFrame(
            [input_data]
        )

    elif isinstance(input_data, pd.Series):

        input_data = input_data.to_frame().T

    elif not isinstance(input_data, pd.DataFrame):

        raise TypeError(
            "input_data must be a pandas "
            "DataFrame, Series, or dictionary."
        )


    
    # CHECK REQUIRED FEATURES
    

    missing_features = [
        c
        for c in EXPECTED_FEATURES
        if c not in input_data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing input features: "
            + str(missing_features)
        )


    
    # SAME FEATURE ORDER AS TRAINING
    

    X = input_data[
        EXPECTED_FEATURES
    ].copy()


    
    # BOOLEAN → INTEGER
    

    bool_cols = feature_info[
        'boolean_features_converted_to_int'
    ]

    existing_bool_cols = [
        c
        for c in bool_cols
        if c in X.columns
    ]

    X[existing_bool_cols] = (
        X[existing_bool_cols]
        .astype(int)
    )


    
    # SAME PREPROCESSING
    

    X_proc = preprocessor.transform(
        X
    )


    
    # XGBOOST PROBABILITY
    

    probability = (
        model.predict_proba(X_proc)[:, 1]
    )


    
    # SAME THRESHOLD
    

    prediction = (
        probability >= THRESHOLD
    ).astype(int)


    
    # FINAL LABEL
    

    detection = np.where(
        prediction == 1,
        'Leak',
        'No Leak'
    )


    return pd.DataFrame(
        {
            'Leak_Probability': probability,
            'Prediction': prediction,
            'Detection': detection
        }
    )
