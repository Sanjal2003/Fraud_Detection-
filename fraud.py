# prompt: generate a streamlit code 

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set Streamlit page config
st.set_page_config(layout="wide")

# Load the trained XGBoost model
try:
    with open('xgb_top_13.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'xgb_top_13.pkl' not found. Please ensure the model is trained and saved.")
    model = None

# Define the required features based on the training data
required_features = [
    'V14', 'V4', 'V1', 'AmountBin', 'V6', 'V26', 'V10', 'V22', 'V21',
    'V17', 'V9', 'V12', 'V25'
]

# Assume the same scalers and label encoders were used as in the notebook
# In a real application, you would save and load these as well.
# For this example, we'll recreate simple placeholder transformers.
# You should replace these with your actual saved transformers.
scaler = StandardScaler()
label_encoders = {
    #'TimeBin': LabelEncoder(),
    'AmountBin': LabelEncoder()
}

# Fit dummy data to the transformers to make them work
# This is a placeholder. In a real app, you'd fit on your training data.
#dummy_time_bins = ['Night', 'Morning', 'Afternoon', 'Evening']
dummy_amount_bins = ['Low', 'Medium', 'High', 'Very High']
#label_encoders['TimeBin'].fit(dummy_time_bins)
label_encoders['AmountBin'].fit(dummy_amount_bins)
# Create a dummy array for scaler fitting (e.g., based on mean/std of training data)
# This is crucial for correct scaling in inference
# Replace with actual values from your training data
dummy_numerical_data = np.array([[0] * (len(required_features) - len(label_encoders.keys()))])
scaler.fit(dummy_numerical_data) # Fit with dummy data


st.title("Credit Card Fraud Detection")
st.markdown("""
    Enter the transaction details below to predict if it's a fraudulent transaction.
""")

# Create input fields for each feature
st.header("Enter Transaction Details")

input_data = {}

# Define ranges or types for each feature for better input widgets
feature_info = {
    'V14': {'type': 'number', 'min_value': -30.0, 'max_value': 10.0},
    'V4': {'type': 'number', 'min_value': -5.0, 'max_value': 10.0},
    'V1': {'type': 'number', 'min_value': -10.0, 'max_value': 10.0},
    'AmountBin': {'type': 'selectbox', 'options': dummy_amount_bins},
    'V6': {'type': 'number', 'min_value': -10.0, 'max_value': 10.0},
    'V26': {'type': 'number', 'min_value': -15.0, 'max_value': 10.0},
    'V10': {'type': 'number', 'min_value': -20.0, 'max_value': 10.0},
    'V22': {'type': 'number', 'min_value': -5.0, 'max_value': 10.0},
    'V21': {'type': 'number', 'min_value': -5.0, 'max_value': 10.0},
    'V17': {'type': 'number', 'min_value': -30.0, 'max_value': 10.0},
    'V9': {'type': 'number', 'min_value': -15.0, 'max_value': 10.0},
    'V12': {'type': 'number', 'min_value': -20.0, 'max_value': 10.0},
    'V25': {'type': 'number', 'min_value': -40.0, 'max_value': 10.0},
}

# Organize inputs into columns
num_cols = 3
cols = st.columns(num_cols)

i = 0
for feature in required_features:
    col = cols[i % num_cols]
    info = feature_info.get(feature, {'type': 'number'}) # Default to number input

    if info['type'] == 'number':
        input_data[feature] = col.number_input(
            f"Enter {feature}",
            min_value=info.get('min_value', None),
            max_value=info.get('max_value', None),
            step=info.get('step', None),
            value=0.0 # Default value
        )
    elif info['type'] == 'selectbox':
         input_data[feature] = col.selectbox(
            f"Select {feature}",
            options=info.get('options', [])
        )
    i += 1


if st.button("Predict Fraud"):
    if model is not None:
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Apply the same preprocessing steps as used during training

        # 1. Handle categorical features (Label Encoding)
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                # Handle unseen labels gracefully if necessary
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError as e:
                    st.warning(f"Warning: Could not transform input for {col}. Possible unseen label.")
                    # Handle the unseen label - e.g., assign a default value or skip
                    # For simplicity here, we'll just note the warning.
                    pass # Decide on how to handle unseen labels

        # 2. Handle numerical features (Scaling)
        numerical_input_cols = [col for col in required_features if col not in label_encoders.keys()]
        if numerical_input_cols:
             # Ensure the order of columns matches the training data if the scaler was fitted on a specific order
             # A more robust approach is to save the list of numerical features used for fitting.
             # For this example, we assume the order in required_features (excluding TimeBin, AmountBin) is correct.
             input_df[numerical_input_cols] = scaler.transform(input_df[numerical_input_cols])


        # Ensure the input DataFrame has the exact same columns in the same order as the training data
        # If features were engineered (like TimeBin, AmountBin) these need to be created BEFORE selection
        # However, in the notebook, TimeBin and AmountBin were created and then label encoded.
        # The `required_features` list already contains the *processed* feature names used by the model.
        # The input process above collects values for these *processed* features directly.

        # Final check: ensure the input DataFrame has the required columns in the correct order
        try:
             input_df_ordered = input_df[required_features]
        except KeyError as e:
            st.error(f"Missing input feature(s): {e}. Please ensure all required fields are provided.")
            st.stop()


        # Make prediction
        prediction = model.predict(input_df_ordered)
        prediction_proba = model.predict_proba(input_df_ordered)[:, 1] # Probability of the positive class (fraud)

        st.header("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**Fraudulent Transaction Detected!**")
        else:
            st.success(f"**Transaction is Likely Not Fraudulent.**")

        st.write(f"Confidence (Probability of Fraud): **{prediction_proba[0]:.4f}**")

    else:
        st.warning("Model is not loaded. Cannot make a prediction.")

