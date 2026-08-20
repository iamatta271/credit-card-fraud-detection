# Credit Card Fraud Detection — Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning.

The project covers data loading, data cleaning, duplicate removal, infinite-value checking, exploratory data analysis, class-distribution analysis, feature preprocessing, feature scaling, model training, prediction, and model evaluation.

The main goal is to build a machine-learning model that can distinguish between normal and fraudulent credit card transactions.

## Project Objectives

* Clean and prepare the credit card transaction dataset.
* Check and handle data-quality issues.
* Identify and remove duplicate transactions.
* Check for missing and infinite values.
* Analyze the distribution of normal and fraudulent transactions.
* Separate features and target variables.
* Split the dataset into training and testing sets.
* Scale numerical features for machine learning.
* Train a Random Forest classification model.
* Make predictions on unseen transactions.
* Evaluate model performance using suitable classification metrics.
* Save the trained model and scaler for future use.

## Dataset

The dataset contains credit card transaction information used to identify fraudulent transactions.

The `Class` column is the target variable:

* `0` = Normal transaction
* `1` = Fraudulent transaction

### Dataset After Cleaning

After removing duplicate rows, the dataset contains:

* **Rows:** 283,726
* **Columns:** 31
* **Features:** 30
* **Target:** `Class`
* **Duplicate Rows Removed:** 1,081
* **Infinite Values:** 0

### Class Distribution

The dataset is highly imbalanced.

| Class | Transaction Type |   Count | Percentage |
| ----- | ---------------- | ------: | ---------: |
| 0     | Normal           | 283,253 |  99.83329% |
| 1     | Fraudulent       |     473 |   0.16671% |

This imbalance is an important part of the project because fraud transactions represent only a very small percentage of all transactions.

## Data Cleaning

The dataset was checked for common data-quality problems.

The following checks were performed:

* Missing values
* Duplicate rows
* Infinite values
* Data types
* Target-class distribution

### Infinite Values

All numerical columns were checked for infinite values.

The result was:

```text
Total Infinite Values: 0
```

This confirms that there were no `inf` or `-inf` values in the numerical columns.

### Duplicate Rows

Duplicate transactions were checked and removed.

Before cleaning:

```text
Duplicate Rows Before Cleaning: 1081
```

After cleaning:

```text
Duplicate Rows After Cleaning: 0
```

The final cleaned dataset contains:

```text
283726 rows
31 columns
```

## Exploratory Data Analysis

Exploratory data analysis was performed to better understand the transaction data and the distribution of fraudulent transactions.

The analysis focuses on:

* Transaction features
* Feature distributions
* Class distribution
* Normal vs fraudulent transactions
* Feature relationships
* Data imbalance

The class distribution shows a significant difference between normal and fraudulent transactions.

## Feature and Target Separation

The `Class` column was separated from the remaining transaction features.

* `X` contains the **30 input features**.
* `y` contains the **target variable (`Class`)**.

Final shapes:

```text
Features Shape: (283726, 30)
Target Shape: (283726,)
```

## Train-Test Split

The dataset was divided into training and testing sets.

An **80/20 split** was used.

* **80%** → Training data
* **20%** → Testing data

The `stratify=y` option was used to maintain the same class distribution in both datasets.

### Training Data

```text
Training Features Shape: (226980, 30)
Training Target Shape: (226980,)
```

### Testing Data

```text
Testing Features Shape: (56746, 30)
Testing Target Shape: (56746,)
```

### Training Class Distribution

| Class |   Count | Percentage |
| ----- | ------: | ---------: |
| 0     | 226,602 | 99.833466% |
| 1     |     378 |  0.166534% |

### Testing Class Distribution

| Class |  Count | Percentage |
| ----- | -----: | ---------: |
| 0     | 56,651 | 99.832587% |
| 1     |     95 |  0.167413% |

The similar percentages confirm that the class distribution was preserved during the train-test split.

## Feature Scaling

Feature scaling was performed using `StandardScaler` from Scikit-learn.

The training data was fitted and transformed using the scaler, while the testing data was only transformed.

This prevents information from the testing dataset from being used during the training process.

The scaling process completed successfully:

```text
Training data scaled successfully
Testing data scaled successfully
```

## Machine Learning Model

A **Random Forest Classifier** was used for fraud detection.

Random Forest is an ensemble machine-learning algorithm that combines multiple decision trees to make predictions.

It is useful for classification problems and can capture complex relationships between different transaction features.

The trained model is saved in the `models` folder.

## Model Files

The trained machine-learning files are stored in the `models` directory:

```text
models/
├── fraud_detection_random_forest.pkl
└── fraud_detection_scaler.pkl
```

### `fraud_detection_random_forest.pkl`

Contains the trained Random Forest model.

### `fraud_detection_scaler.pkl`

Contains the StandardScaler used to preprocess the transaction features.

Saving both files allows the trained model to be reused later without training it again.

## Model Evaluation

Because the dataset is highly imbalanced, accuracy alone is not enough to evaluate a fraud detection model.

The model should be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Why These Metrics Matter

**Precision** measures how many transactions predicted as fraud were actually fraudulent.

**Recall** measures how many actual fraudulent transactions were successfully detected.

**F1-score** provides a balance between precision and recall.

The **confusion matrix** shows:

* True Negatives
* False Positives
* False Negatives
* True Positives

For fraud detection, recall is particularly important because missing an actual fraudulent transaction can be costly.

## Project Structure

```text
credit-card-fraud-detection/
│
├── notebook/
│   └── credit_card_fraud_detection.ipynb
│
├── models/
│   ├── fraud_detection_random_forest.pkl
│   └── fraud_detection_scaler.pkl
│
├── README.md
└── .gitignore
```

## Tools and Libraries

The project was developed using Python and the following libraries:

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Project Workflow

The complete workflow is:

```text
Data Loading
     ↓
Data Cleaning
     ↓
Check Missing Values
     ↓
Check Infinite Values
     ↓
Remove Duplicate Rows
     ↓
Analyze Class Distribution
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Check Class Distribution
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Prediction
     ↓
Model Evaluation
     ↓
Save Model and Scaler
```

## Repository Contents

* `credit_card_fraud_detection.ipynb` — Complete fraud detection notebook
* `fraud_detection_random_forest.pkl` — Trained Random Forest model
* `fraud_detection_scaler.pkl` — Feature scaler
* `README.md` — Project documentation
* `.gitignore` — Files ignored by Git

## Conclusion

This project demonstrates how Machine Learning can be used to identify fraudulent credit card transactions.

The dataset was cleaned by removing **1,081 duplicate rows**, and no infinite values were found.

After cleaning, the dataset contained **283,726 transactions and 30 input features**. The dataset was highly imbalanced, with only **473 fraudulent transactions**, representing approximately **0.17%** of the data.

The data was split into training and testing sets while maintaining the original class distribution. The features were then scaled and used to train a Random Forest classification model.

The trained model and scaler were saved so they can be reused for future fraud predictions.

