Gait analysis is the study of human walking patterns using measurements such as joint angles, movement characteristics, and timing information.

In this project, gait angle time-series data is processed and transformed into meaningful statistical and temporal features. These engineered features are then used by Machine Learning algorithms to classify different gait conditions.

##  Objectives

The main objectives of this project are:

- Analyze gait angle time-series data.
- Preprocess and organize gait measurements.
- Extract meaningful biomechanical features.
- Convert raw time-series data into structured feature vectors.
- Compare different Machine Learning algorithms.
- Evaluate model performance using subject-wise validation.
- Train and save a final Machine Learning model.
- Develop an interactive Streamlit application.
- Allow users to upload gait CSV files and obtain predictions.
- Visualize gait angle data and prediction results.

---

## Dataset

The dataset consists of gait time-series measurements collected under different brace conditions.

Each observation contains information related to:

- `subject` – subject identifier
- `condition` – gait condition/class
- `replication` – repeated gait trial
- `leg` – leg identifier
- `joint` – joint identifier
- `time` – time point in the gait cycle
- `angle` – measured joint angle

### Dataset Information

| Parameter | Value |
|-----------|------:|
| Total observations | 181,800 |
| Subjects | 10 |
| Conditions | 3 |
| Legs | 2 |
| Joints | 3 |

### Target Classes

- **Class 1:** Unbraced
- **Class 2:** Knee Brace
- **Class 3:** Ankle Brace

The three target classes are balanced in the dataset.

---

## Data Preprocessing

The raw gait data is organized according to:

```text
Subject
Replication
Leg
Joint
Time
Angle
````

The data is grouped by:

```text
subject
replication
leg
joint
```

This allows the time-series measurements of each gait recording to be summarized into representative features.

---

## Feature Engineering

Raw gait time-series data contains many individual angle measurements. Instead of directly using every time point, statistical and temporal characteristics are extracted from each gait recording.

For every:

```text
Subject + Replication + Leg + Joint
```

the following features are calculated.

### Statistical Features

* Mean
* Standard Deviation
* Minimum
* Maximum
* Range

### Temporal Feature

* Time of Peak

The range is calculated as:

```text
Range = Maximum - Minimum
```

The time of peak represents the point in the gait cycle where the joint angle reaches its maximum value.

---

## Final Feature Set

Features are generated for:

* 2 legs
* 3 joints
* 4 final feature groups

The four feature groups used for the final model are:

```text
Mean
Standard Deviation
Range
Time of Peak
```

Therefore, the final feature representation contains:

```text
2 × 3 × 4 = 24 features
```

These 24 engineered gait features are used as input to the Machine Learning models.

---

## Machine Learning Models

Three Machine Learning algorithms were evaluated during the project.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification algorithm.

It provides a simple linear approach for separating the three gait conditions.

### 2. Random Forest

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees.

The Random Forest model used in this project contains:

```text
n_estimators = 100
random_state = 42
```

### 3. XGBoost

XGBoost was also evaluated as a gradient boosting-based classification algorithm.

It was included to compare the performance of a boosting approach with Logistic Regression and Random Forest.

---

## Model Performance

The initial train-test evaluation produced the following results:

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |     100% |
| Random Forest       |     100% |
| XGBoost             |     100% |

All three models achieved 100% accuracy on the recorded train-test evaluation.

---

## Subject-Wise Validation

Because gait data contains repeated measurements from the same subjects, subject-level data leakage can be an important concern.

To address this, subject-wise validation was performed using **Leave-One-Subject-Out (LOSO) cross-validation**.

In LOSO validation:

1. One subject is completely held out.
2. The model is trained on the remaining subjects.
3. The held-out subject is used for testing.
4. The process is repeated for all subjects.

### LOSO Results

```text
Average Accuracy: 100%
Standard Deviation: 0%
```

All 10 subjects achieved 100% accuracy in the recorded LOSO experiment.

> ⚠️ These results represent performance on the available project dataset. They should not be interpreted as clinical validation or proof of diagnostic performance on unseen real-world patient populations.

---

## Final Model

For deployment, a Random Forest model was selected.

The final prediction pipeline consists of:

```text
Input Gait Features
        ↓
StandardScaler
        ↓
Random Forest Classifier
        ↓
Predicted Gait Condition
```

The trained pipeline is saved as:

```text
gait_pipeline.pkl
```

The model feature names are saved as:

```text
gait_features.pkl
```

These files are loaded by the Streamlit application during prediction.

---

## Streamlit Application

An interactive Streamlit application was developed to make the trained Machine Learning model accessible through a web interface.

### The application allows users to:

* 📂 Upload a gait CSV file
* 📊 Preview the uploaded data
* ✅ Check required columns
* 🔧 Automatically perform feature engineering
* 🧬 Generate the required 24 gait features
* 🤖 Run the trained Random Forest model
* 🎯 Display the predicted gait condition
* 📈 Display prediction probabilities
* 📉 Visualize gait angle data
* 🔍 Explore predictions for uploaded recordings

---

## 🔄 Application Workflow

```text
                  Raw Gait CSV
                       │
                       ▼
               Data Validation
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
         ┌────────────────────────────┐
         │ Mean                       │
         │ Standard Deviation         │
         │ Range                      │
         │ Time of Peak               │
         └────────────────────────────┘
                       │
                       ▼
               24 Gait Features
                       │
                       ▼
                 StandardScaler
                       │
                       ▼
               Random Forest Model
                       │
                       ▼
                   Prediction
                       │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
          Unbraced  Knee Brace  Ankle Brace
```

---

## Gait Visualization

The Streamlit application also provides visualization of the uploaded gait angle data.

Users can select:

* Subject
* Leg
* Joint

and view the corresponding joint-angle measurements over time.

This provides a visual representation of the gait pattern alongside the Machine Learning prediction.

---

## Model Explainability

Model interpretability was also explored using **SHAP (SHapley Additive exPlanations)**.

SHAP was used to investigate the contribution and importance of engineered gait features for the Random Forest model.

This provides additional insight into which gait characteristics contribute to the classification results.

---

## Project Structure

```text
multivariate-gaitanalysis/
│
├── app.py
├── gait_pipeline.pkl
├── gait_features.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                | Description                    |
| ------------------- | ------------------------------ |
| `app.py`            | Streamlit web application      |
| `gait_pipeline.pkl` | Trained Random Forest pipeline |
| `gait_features.pkl` | List of model feature names    |
| `requirements.txt`  | Required Python libraries      |
| `README.md`         | Project documentation          |

> 📌 The gait CSV dataset is maintained separately and uploaded by the user through the Streamlit application.

---

## Run Locally

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## Requirements

The project uses the following Python libraries:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## Deployment

The Streamlit application is deployed through **Streamlit Community Cloud** using the GitHub repository.

### Deployment Workflow

```text
Google Colab
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Saving
      ↓
GitHub Repository
      ↓
Streamlit Community Cloud
      ↓
Live Gait Analysis Application
```

---

## Example Input

A labeled dataset may contain:

```csv
subject,condition,replication,leg,joint,time,angle
1,1,1,1,1,0,5.23
1,1,1,1,1,1,5.41
1,1,1,1,1,2,5.72
1,1,1,1,1,3,6.01
```

For prediction using unlabeled gait data:

```csv
subject,replication,leg,joint,time,angle
1,1,1,1,0,5.23
1,1,1,1,1,5.41
1,1,1,1,2,5.72
1,1,1,1,3,6.01
```

The application automatically processes the uploaded data and generates the required features.

---

## Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* XGBoost

### Explainable AI

* SHAP

### Visualization

* Matplotlib
* Streamlit

### Development Environment

* Google Colab

### Deployment

* GitHub
* Streamlit Community Cloud

---


## Project Link

| Resource     | Link                                            |
| ------------ | ----------------------------------------------- |
| 🌐 Live Demo | [Streamlit App] : https://multivariate-gaitanalysis-hghbyfufwpsr5s5moygddc.streamlit.app/ |


---

