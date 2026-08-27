import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Gait Analysis AI",
    page_icon="🦵",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #ddd;
    margin-top: 20px;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
}

.result-text {
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🦵 AI-Based Gait Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Classification of Gait Conditions'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL FILES
# =========================================================

@st.cache_resource
def load_model():

    pipeline = joblib.load("gait_pipeline.pkl")
    feature_names = joblib.load("gait_features.pkl")

    return pipeline, feature_names


try:

    model, feature_names = load_model()

except FileNotFoundError as e:

    st.error(
        "❌ Model files are missing. "
        "Make sure the following files are in your GitHub repository:"
    )

    st.code("""
gait_pipeline.pkl
gait_features.pkl
    """)

    st.stop()

except Exception as e:

    st.error("❌ Error loading model files.")
    st.exception(e)
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📌 About the App")

st.sidebar.write(
    """
This application uses a trained Random Forest machine learning
model to classify gait patterns.

The model analyzes engineered gait features obtained from
joint angle time-series data.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("Classification Classes")

st.sidebar.write("**1 → Unbraced**")
st.sidebar.write("**2 → Knee Brace**")
st.sidebar.write("**3 → Ankle Brace**")

st.sidebar.markdown("---")

st.sidebar.subheader("Expected CSV Columns")

st.sidebar.code("""
subject
condition
replication
leg
joint
time
angle
""")

st.sidebar.info(
    "For prediction, the condition column is not required. "
    "If present, it is ignored by the prediction pipeline."
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.header("📂 Upload Gait CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing gait time-series data",
    type=["csv"]
)


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # READ CSV
        # -------------------------------------------------

        df = pd.read_csv(uploaded_file)

        st.success("✅ CSV uploaded successfully!")

        # -------------------------------------------------
        # DATA PREVIEW
        # -------------------------------------------------

        st.subheader("📊 Uploaded Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", f"{len(df):,}")

        with col2:
            st.metric("Columns", len(df.columns))

        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("View first 10 rows"):

            st.dataframe(
                df.head(10),
                use_container_width=True
            )


        # =================================================
        # CHECK REQUIRED COLUMNS
        # =================================================

        required_columns = [
            "subject",
            "replication",
            "leg",
            "joint",
            "time",
            "angle"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                "❌ The uploaded CSV is missing required columns:"
            )

            st.write(missing_columns)

            st.info(
                "Required columns are: "
                + ", ".join(required_columns)
            )

            st.stop()


        # =================================================
        # CONVERT NUMERIC COLUMNS
        # =================================================

        numeric_columns = [
            "subject",
            "replication",
            "leg",
            "joint",
            "time",
            "angle"
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        # Remove rows with invalid required values

        df = df.dropna(
            subset=numeric_columns
        ).copy()


        if len(df) == 0:

            st.error(
                "❌ No valid data remains after cleaning."
            )

            st.stop()


        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        st.header("🔧 Feature Engineering")

        st.write(
            "The uploaded gait time-series is converted into "
            "statistical gait features."
        )


        # -------------------------------------------------
        # GROUP DATA
        # -------------------------------------------------
        #
        # Same basic feature engineering used in Colab:
        #
        # subject
        # replication
        # leg
        # joint
        #
        # condition is NOT required for prediction because
        # it is the target we want to predict.
        #
        # -------------------------------------------------

        grouped = df.groupby(
            [
                "subject",
                "replication",
                "leg",
                "joint"
            ]
        )["angle"]


        # -------------------------------------------------
        # STATISTICAL FEATURES
        # -------------------------------------------------

        features = grouped.agg(
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        ).reset_index()


        # Range = max - min

        features["range"] = (
            features["max"] -
            features["min"]
        )


        # -------------------------------------------------
        # TIME OF PEAK
        # -------------------------------------------------

        peak_indices = df.groupby(
            [
                "subject",
                "replication",
                "leg",
                "joint"
            ]
        )["angle"].idxmax()


        peak_time = df.loc[
            peak_indices
        ][
            [
                "subject",
                "replication",
                "leg",
                "joint",
                "time"
            ]
        ].rename(
            columns={
                "time": "time_of_peak"
            }
        )


        # Merge peak time with features

        features = features.merge(
            peak_time,
            on=[
                "subject",
                "replication",
                "leg",
                "joint"
            ],
            how="left"
        )


        # =================================================
        # PIVOT FEATURES
        # =================================================

        pivoted = features.pivot_table(
            index=[
                "subject",
                "replication"
            ],
            columns=[
                "leg",
                "joint"
            ],
            values=[
                "mean",
                "std",
                "range",
                "time_of_peak"
            ]
        )


        # Flatten MultiIndex columns

        pivoted.columns = [
            "_".join(
                map(str, col)
            )
            for col in pivoted.columns
        ]


        pivoted = pivoted.reset_index()


        # =================================================
        # REMOVE ID COLUMNS
        # =================================================

        X = pivoted.drop(
            columns=[
                "subject",
                "replication"
            ],
            errors="ignore"
        )


        # =================================================
        # ALIGN FEATURES WITH TRAINING FEATURES
        # =================================================

        missing_features = [
            feature
            for feature in feature_names
            if feature not in X.columns
        ]

        extra_features = [
            feature
            for feature in X.columns
            if feature not in feature_names
        ]


        # -------------------------------------------------
        # CHECK MISSING FEATURES
        # -------------------------------------------------

        if missing_features:

            st.error(
                "❌ The uploaded gait data does not contain "
                "enough information to generate all features "
                "required by the trained model."
            )

            st.write(
                "Missing engineered features:"
            )

            st.code(
                "\n".join(missing_features)
            )

            st.info(
                "Make sure your CSV contains both legs "
                "and all three joints with their time-series angles."
            )

            st.stop()


        # -------------------------------------------------
        # KEEP EXACT TRAINING FEATURE ORDER
        # -------------------------------------------------

        X = X[
            feature_names
        ]


        # -------------------------------------------------
        # HANDLE MISSING VALUES
        # -------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(
            X.median()
        )


        # =================================================
        # SHOW ENGINEERED FEATURES
        # =================================================

        with st.expander(
            "🔍 View engineered features"
        ):

            st.write(
                f"Generated {X.shape[1]} model features."
            )

            st.dataframe(
                X,
                use_container_width=True
            )


        # =================================================
        # PREDICTION BUTTON
        # =================================================

        st.markdown("---")

        predict_button = st.button(
            "🔮 Predict Gait Condition",
            type="primary",
            use_container_width=True
        )


        if predict_button:

            # =============================================
            # MODEL PREDICTION
            # =============================================

            predictions = model.predict(X)


            # Probability if supported

            try:

                probabilities = model.predict_proba(X)

            except Exception:

                probabilities = None


            # =============================================
            # LABEL MAPPING
            # =============================================

            label_map = {
                1: "Unbraced",
                2: "Knee Brace",
                3: "Ankle Brace"
            }


            # =============================================
            # DISPLAY RESULTS
            # =============================================

            st.header("🧠 Prediction Result")


            # If multiple gait recordings exist

            result_df = pd.DataFrame({
                "Subject": pivoted["subject"],
                "Replication": pivoted["replication"],
                "Predicted Class": predictions,
                "Predicted Condition": [
                    label_map.get(
                        int(pred),
                        f"Class {pred}"
                    )
                    for pred in predictions
                ]
            })


            # =============================================
            # SINGLE / MULTIPLE RESULTS
            # =============================================

            if len(result_df) == 1:

                prediction = int(
                    predictions[0]
                )

                condition_name = label_map.get(
                    prediction,
                    f"Class {prediction}"
                )


                st.markdown(
                    f"""
                    <div class="result-box">

                    <div class="result-title">
                    {condition_name}
                    </div>

                    <div class="result-text">
                    Predicted Gait Condition
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # -----------------------------------------
                # PROBABILITY
                # -----------------------------------------

                if probabilities is not None:

                    st.subheader(
                        "📈 Prediction Confidence"
                    )

                    classes = model.classes_

                    probability_df = pd.DataFrame({
                        "Condition": [
                            label_map.get(
                                int(c),
                                f"Class {c}"
                            )
                            for c in classes
                        ],
                        "Probability": probabilities[0]
                    })

                    probability_df[
                        "Probability"
                    ] = (
                        probability_df[
                            "Probability"
                        ] * 100
                    ).round(2)

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.bar_chart(
                        probability_df.set_index(
                            "Condition"
                        )["Probability"]
                    )


            else:

                st.subheader(
                    "📋 Predictions for Uploaded Recordings"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )


                # -----------------------------------------
                # OVERALL PREDICTION DISTRIBUTION
                # -----------------------------------------

                result_counts = (
                    result_df[
                        "Predicted Condition"
                    ]
                    .value_counts()
                )

                st.subheader(
                    "📊 Prediction Distribution"
                )

                st.bar_chart(
                    result_counts
                )


            # =================================================
            # GAIT ANGLE VISUALIZATION
            # =================================================

            st.markdown("---")

            st.header(
                "📉 Gait Angle Visualization"
            )

            st.write(
                "Raw joint-angle measurements from the "
                "uploaded gait recording."
            )


            # Select subject

            subjects = sorted(
                df["subject"]
                .dropna()
                .unique()
            )

            if len(subjects) > 0:

                selected_subject = st.selectbox(
                    "Select Subject",
                    subjects
                )

                subject_data = df[
                    df["subject"] ==
                    selected_subject
                ].copy()


                # Select leg

                legs = sorted(
                    subject_data["leg"]
                    .dropna()
                    .unique()
                )

                if len(legs) > 0:

                    selected_leg = st.selectbox(
                        "Select Leg",
                        legs
                    )

                    leg_data = subject_data[
                        subject_data["leg"] ==
                        selected_leg
                    ].copy()


                    # Select joint

                    joints = sorted(
                        leg_data["joint"]
                        .dropna()
                        .unique()
                    )

                    if len(joints) > 0:

                        selected_joint = st.selectbox(
                            "Select Joint",
                            joints
                        )

                        plot_data = leg_data[
                            leg_data["joint"] ==
                            selected_joint
                        ].sort_values(
                            "time"
                        )


                        if len(plot_data) > 0:

                            chart_data = (
                                plot_data[
                                    [
                                        "time",
                                        "angle"
                                    ]
                                ]
                                .set_index("time")
                            )

                            st.line_chart(
                                chart_data
                            )


            # =================================================
            # IMPORTANT DISCLAIMER
            # =================================================

            st.markdown("---")

            st.warning(
                "⚠️ This application is a machine-learning "
                "demonstration based on gait data. Predictions "
                "should not be considered a clinical diagnosis."
            )


    except Exception as e:

        st.error(
            "❌ Something went wrong while processing the CSV."
        )

        st.exception(e)

else:

    # =====================================================
    # BEFORE FILE UPLOAD
    # =====================================================

    st.info(
        "👆 Upload a gait CSV file above to start the analysis."
    )

    st.markdown("---")

    st.subheader(
        "How it works"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            ### 1️⃣ Upload
            Upload raw gait time-series CSV data.
            """
        )

    with col2:

        st.markdown(
            """
            ### 2️⃣ Features
            Statistical gait features are automatically
            extracted.
            """
        )

    with col3:

        st.markdown(
            """
            ### 3️⃣ AI Model
            The trained Random Forest model analyzes
            the gait features.
            """
        )

    with col4:

        st.markdown(
            """
            ### 4️⃣ Result
            The predicted gait condition is displayed.
            """
        )
