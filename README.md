Fraud Detection Model for Financial Transactions
Objective:
Developed a machine learning model to detect fraudulent transactions in financial datasets, aiming to identify anomalies and reduce financial loss in real-time systems.

Dataset:
Credit card transaction data from Kaggle, including features like v1,v2,...,28, transaction amount, time, and class labels (fraud/non-fraud).

Key Steps:

Exploratory Data Analysis (EDA):
Analyzed class imbalance, transaction patterns, and feature distributions.
Scatter plot for time vs amount.
Visualized correlations and identified behavior-based anomalies.

Data Preprocessing:
Handled missing values and outliers.
Scaled continuous variables.
Managed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

Feature Engineering:
Created time-based features (hour, Timebin, IsNight, DeltaTimePrev).
Engineered amount-based features (LogAmount, HighAmountFlag, AmountBin, ZScoreAmount).
Included behavioral based features (TxnCountPastHour, CumulativeTxnAmount)

Model Training:
Implemented and compared multiple models:
Logistic Regression, Decision Tree, XGBoost.
Evaluated using F1-score, Precision, Recall, Confusion matrix and ROC-AUC.

Model Optimization:
Used feature importance to refine and reduce the feature set.
Implemented XGBoost model with refined top 13 features.

Deployment:
Developed a real-time prediction pipeline.
Created an interactive Streamlit dashboard for model input/output and fraud alerts.

Tech Stack:
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, Jupyter Notebook
