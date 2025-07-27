import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Income Predictor",
    page_icon="🧑💻",
    layout="wide"
)

# --- Load Model and Artifacts ---
@st.cache_resource
def load_assets():
    models_dir = 'models'
    try:
        model = joblib.load(os.path.join(models_dir, 'salary_prediction_model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        training_columns = joblib.load(os.path.join(models_dir, 'training_columns.pkl'))
        st.sidebar.success("Model and components loaded successfully!")
        return model, scaler, label_encoders, training_columns
    except Exception as e:
        st.sidebar.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

model, scaler, label_encoders, training_columns = load_assets()

# --- User Input UI ---
st.title("Employee Income Prediction System 🧑💻")
st.markdown("---")

if model is None:
    st.error("Model and required assets are not loaded. Please upload .pkl files in the 'models' folder.")
    st.stop()

# Define function to get user inputs
def get_user_input():
    # Numeric inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, max_value=1000000, value=100000)
    education_num = st.number_input("Education Number", min_value=1, max_value=20, value=10)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=1000000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=1000000, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=168, value=40)
    
    # Categorical inputs
    workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                         'State-gov', 'Without-pay', 'Never-worked']  # Adapt based on training categories
    workclass = st.selectbox("Workclass", options=workclass_options)
    
    education_options = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                         'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                         '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
    education = st.selectbox("Education", options=education_options)
    
    marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                              'Married-spouse-absent', 'Married-AF-spouse']
    marital_status = st.selectbox("Marital Status", marital_status_options)
    
    occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                          'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                          'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                          'Armed-Forces']
    occupation = st.selectbox("Occupation", occupation_options)
    
    relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    relationship = st.selectbox("Relationship", relationship_options)
    
    race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    race = st.selectbox("Race", race_options)
    
    gender_options = ['Male', 'Female']
    gender = st.selectbox("Gender", gender_options)
    
    native_country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                              'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                              'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
                              'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
                              'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                              'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                              'Peru', 'Hong', 'Holand-Netherlands']
    native_country = st.selectbox("Native Country", native_country_options)
    
    # Package into dict
    user_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    return user_data

user_input = get_user_input()

# --- Preprocess & Predict ---
if st.button("Predict Income"):
    try:
        # Encode categorical inputs using saved label_encoders
        user_input_encoded = user_input.copy()
        for col, le in label_encoders.items():
            if col in user_input_encoded:
                # Convert single value into list for transform, then extract back the value
                user_input_encoded[col] = le.transform([user_input_encoded[col]])[0]

        # Convert dict to df for correct column order
        user_input_df = pd.DataFrame([user_input_encoded])
        
        # Reindex columns to training_columns (adds missing cols as NaN)
        user_input_df = user_input_df.reindex(columns=training_columns, fill_value=0)

        # Apply scaler
        user_input_scaled = scaler.transform(user_input_df)
        
        # Predict
        prediction = model.predict(user_input_scaled)
        
        # Map prediction to readable label if required
        pred_label = "<=50K" if prediction[0] == 0 else ">50K"
        
        st.success(f"Predicted Income: {pred_label}")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

st.markdown("---")
st.write("Enter the employee details on the left and click 'Predict Income' to see the result.")
