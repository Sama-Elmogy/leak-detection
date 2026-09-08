import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import xgboost as xgb


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Leak Detection System",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(BASE_DIR, "xgboost_artifacts")

MODEL_PATH = os.path.join(A, "xgboost_model.ubj")
PREPROCESSOR_PATH = os.path.join(A, "preprocessor.pkl")
FEATURE_INFO_PATH = os.path.join(A, "feature_info.json")
THRESHOLD_PATH = os.path.join(A, "threshold.json")
METRICS_PATH = os.path.join(A, "xgboost_metrics.json")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_INFO_PATH,
    THRESHOLD_PATH,
    METRICS_PATH
]

missing_files = [
    path for path in required_files
    if not os.path.isfile(path)
]

if missing_files:

    st.error("❌ Missing model/artifact files.")

    st.write("The following files are missing:")

    for path in missing_files:
        st.write(f"- `{path}`")

    st.stop()


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model_and_preprocessor():

    # Load native XGBoost model
    model = xgb.XGBClassifier()

    model.load_model(MODEL_PATH)

    # Load preprocessing pipeline
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================

try:

    model, preprocessor = load_model_and_preprocessor()

    feature_info = load_json(
        FEATURE_INFO_PATH
    )

    threshold_data = load_json(
        THRESHOLD_PATH
    )

    metrics = load_json(
        METRICS_PATH
    )

except Exception as e:

    st.error(
        "❌ Could not load the saved model/artifacts."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = []


# First choice:
# Get original feature names directly from the fitted preprocessor

try:

    feature_names = list(
        preprocessor.feature_names_in_
    )

except Exception:

    feature_names = []


# Second choice:
# Get feature names from feature_info.json

if not feature_names:

    if isinstance(feature_info, list):

        feature_names = feature_info

    elif isinstance(feature_info, dict):

        possible_keys = [
            "features",
            "feature_names",
            "columns",
            "input_features",
            "all_features",
            "raw_input_features"
        ]

        for key in possible_keys:

            if key in feature_info:

                if isinstance(
                    feature_info[key],
                    list
                ):

                    feature_names = (
                        feature_info[key]
                    )

                    break


# ============================================================
# THRESHOLD
# ============================================================

if isinstance(threshold_data, dict):

    threshold = threshold_data.get(
        "best_threshold",
        threshold_data.get(
            "threshold",
            threshold_data.get(
                "optimal_threshold",
                0.5
            )
        )
    )

else:

    threshold = threshold_data


threshold = float(threshold)


# ============================================================
# METRICS
# ============================================================

def get_metric(name):

    if isinstance(metrics, dict):

        return metrics.get(
            name,
            "N/A"
        )

    return "N/A"


accuracy = get_metric("accuracy")
precision = get_metric("precision")
recall = get_metric("recall")
f1 = get_metric("f1")
roc_auc = get_metric("roc_auc")
pr_auc = get_metric("pr_auc")


# ============================================================
# HEADER
# ============================================================

st.title(
    "💧 Leak Detection System"
)

st.write(
    "XGBoost-based leak detection using the "
    "trained model, saved preprocessing pipeline, "
    "and validation threshold."
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(
    "📊 Model Performance"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Accuracy",
        (
            f"{accuracy:.4f}"
            if isinstance(
                accuracy,
                (int, float)
            )
            else accuracy
        )
    )


with c2:

    st.metric(
        "Precision",
        (
            f"{precision:.4f}"
            if isinstance(
                precision,
                (int, float)
            )
            else precision
        )
    )


with c3:

    st.metric(
        "Recall",
        (
            f"{recall:.4f}"
            if isinstance(
                recall,
                (int, float)
            )
            else recall
        )
    )


with c4:

    st.metric(
        "F1 Score",
        (
            f"{f1:.4f}"
            if isinstance(
                f1,
                (int, float)
            )
            else f1
        )
    )


c5, c6, c7 = st.columns(3)


with c5:

    st.metric(
        "ROC-AUC",
        (
            f"{roc_auc:.4f}"
            if isinstance(
                roc_auc,
                (int, float)
            )
            else roc_auc
        )
    )


with c6:

    st.metric(
        "PR-AUC",
        (
            f"{pr_auc:.4f}"
            if isinstance(
                pr_auc,
                (int, float)
            )
            else pr_auc
        )
    )


with c7:

    st.metric(
        "Threshold",
        f"{threshold:.4f}"
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Model Settings"
)

st.sidebar.write(
    "The model uses the threshold selected "
    "on the validation set."
)

st.sidebar.metric(
    "Saved Threshold",
    f"{threshold:.4f}"
)

st.sidebar.write(
    "Prediction rule:"
)

st.sidebar.code(
    f"Leak if probability >= {threshold:.4f}"
)


# ============================================================
# TABS
# ============================================================

single_tab, batch_tab = st.tabs(
    [
        "🔹 Single Prediction",
        "📁 CSV Batch Prediction"
    ]
)


# ============================================================
# SINGLE PREDICTION
# ============================================================

with single_tab:

    st.header(
        "Single Observation"
    )

    if not feature_names:

        st.error(
            "❌ Could not determine the original feature names."
        )

        st.stop()


    st.write(
        "Enter values for the model input features."
    )


    input_data = {}


    # ========================================================
    # INPUT WIDGETS
    # ========================================================

    for feature in feature_names:

        feature_lower = feature.lower()


        # Detect boolean-like features
        if (
            "bool" in feature_lower
            or feature_lower.startswith("is_")
            or feature_lower.startswith("has_")
        ):

            input_data[feature] = st.checkbox(
                feature,
                value=False
            )

        else:

            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )


    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(
        "🔍 Predict Leak Status",
        type="primary",
        use_container_width=True
    ):

        try:

            # Create dataframe in EXACT training feature order
            input_df = pd.DataFrame(
                [input_data],
                columns=feature_names
            )


            # =================================================
            # BOOLEAN CONVERSION
            # =================================================

            bool_cols = (
                input_df
                .select_dtypes(
                    include="bool"
                )
                .columns
                .tolist()
            )


            if bool_cols:

                input_df[bool_cols] = (
                    input_df[bool_cols]
                    .astype(int)
                )


            # =================================================
            # SAME PREPROCESSING AS TRAINING
            # =================================================

            input_processed = (
                preprocessor.transform(
                    input_df
                )
            )


            # =================================================
            # PREDICT PROBABILITY
            # =================================================

            probability = (
                model.predict_proba(
                    input_processed
                )[0, 1]
            )


            # =================================================
            # APPLY SAVED THRESHOLD
            # =================================================

            prediction = int(
                probability >= threshold
            )


            # =================================================
            # DISPLAY RESULT
            # =================================================

            st.divider()

            st.subheader(
                "Prediction Result"
            )


            r1, r2 = st.columns(2)


            with r1:

                st.metric(
                    "Leak Probability",
                    f"{probability:.2%}"
                )


            with r2:

                if prediction == 1:

                    st.error(
                        "🚨 LEAK DETECTED"
                    )

                else:

                    st.success(
                        "✅ NO LEAK DETECTED"
                    )


            st.write(
                f"Decision threshold: "
                f"**{threshold:.4f}**"
            )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# CSV BATCH PREDICTION
# ============================================================

with batch_tab:

    st.header(
        "CSV Batch Prediction"
    )

    st.write(
        "Upload a CSV containing the same "
        "input features used during model training."
    )


    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            # =================================================
            # READ CSV
            # =================================================

            df = pd.read_csv(
                uploaded_file
            )


            st.subheader(
                "Uploaded Dataset"
            )


            st.write(
                f"Rows: **{len(df):,}**"
            )

            st.write(
                f"Columns: **{len(df.columns):,}**"
            )


            st.dataframe(
                df.head(10),
                use_container_width=True
            )


            # =================================================
            # CHECK MISSING FEATURES
            # =================================================

            missing_columns = [
                col
                for col in feature_names
                if col not in df.columns
            ]


            if missing_columns:

                st.error(
                    "❌ The CSV is missing required features:"
                )


                for col in missing_columns:

                    st.write(
                        f"- `{col}`"
                    )


                st.stop()


            # =================================================
            # SELECT FEATURES
            # IN EXACT TRAINING ORDER
            # =================================================

            X_input = df[
                feature_names
            ].copy()


            # =================================================
            # BOOLEAN CONVERSION
            # =================================================

            bool_cols = (
                X_input
                .select_dtypes(
                    include="bool"
                )
                .columns
                .tolist()
            )


            if bool_cols:

                X_input[bool_cols] = (
                    X_input[bool_cols]
                    .astype(int)
                )


            # =================================================
            # SAME PREPROCESSING AS TRAINING
            # =================================================

            X_processed = (
                preprocessor.transform(
                    X_input
                )
            )


            # =================================================
            # PREDICTION
            # =================================================

            probabilities = (
                model.predict_proba(
                    X_processed
                )[:, 1]
            )


            # =================================================
            # APPLY SAME THRESHOLD
            # =================================================

            predictions = (
                probabilities >= threshold
            ).astype(int)


            # =================================================
            # CREATE RESULTS
            # =================================================

            result_df = df.copy()


            result_df[
                "Leak_Probability"
            ] = probabilities


            result_df[
                "Prediction"
            ] = predictions


            result_df[
                "Detection"
            ] = np.where(
                predictions == 1,
                "Leak",
                "No Leak"
            )


            # =================================================
            # SUMMARY
            # =================================================

            st.divider()

            st.subheader(
                "Prediction Summary"
            )


            total = len(
                result_df
            )


            leak_count = int(
                predictions.sum()
            )


            no_leak_count = (
                total - leak_count
            )


            s1, s2, s3 = st.columns(3)


            with s1:

                st.metric(
                    "Total Observations",
                    f"{total:,}"
                )


            with s2:

                st.metric(
                    "Leaks Detected",
                    f"{leak_count:,}"
                )


            with s3:

                st.metric(
                    "No Leak",
                    f"{no_leak_count:,}"
                )


            # =================================================
            # RESULT TABLE
            # =================================================

            st.subheader(
                "Prediction Results"
            )


            st.dataframe(
                result_df,
                use_container_width=True
            )


            # =================================================
            # DOWNLOAD RESULTS
            # =================================================

            csv_output = (
                result_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )


            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_output,
                file_name="leak_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


        except Exception as e:

            st.error(
                "❌ Batch prediction failed."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Leak Detection System | XGBoost 3.4.1 | "
    "Native UBJ Model | Saved Validation Threshold"
)