# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os
# from sklearn.preprocessing import LabelEncoder

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Income Predictor",
#     page_icon="🧑‍💻",
#     layout="wide"
# )

# # --- UI Styling (Enhanced CSS) ---
# st.markdown(
#     """
#     <style>
#     /* General Styling */
#     .reportview-container, .main {
#         background: #f0f2f6;
#     }
#     /* Title Styling - Hides the anchor link icon */
#     h1 {
#         color: #007BFF;
#         text-align: center;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         font-weight: bold;
#     }
#     h1 a {
#         display: none !important;
#     }
#     h2, h3 { color: #0056b3; }
#     .stButton>button {
#         background-color: #28A745;
#         color: white;
#         border-radius: 8px;
#         padding: 10px 20px;
#         font-size: 16px;
#         font-weight: bold;
#         border: none;
#         box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
#         transition: background-color 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #218838;
#         box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
#     }
#     .sidebar .sidebar-content { background-color: #f8f9fa; }

#     [data-testid="stSidebar"] .stButton > button {
#         background: linear-gradient(90deg, #228f56 0%, #28d790 100%);
#         color: #fff !important;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         font-size: 18px;
#         font-weight: 800;
#         letter-spacing: 0.025em;
#         border: none;
#         border-radius: 30px;
#         text-shadow: 0 2px 13px rgba(20,32,34,0.33);
#         box-shadow: 0 3px 20px 0 rgba(34,143,86,0.14);
#         outline: 2px solid #28a745;
#         outline-offset: 1px;
#         padding: 14px 38px;
#         margin-top: 18px;
#         transition:
#             background 0.18s,
#             box-shadow 0.19s,
#             outline-color 0.13s,
#             transform 0.12s,
#             color 0.12s;
#     }
#     [data-testid="stSidebar"] .stButton > button:hover {
#         background: linear-gradient(90deg, #158349 0%, #27c982 100%);
#         color: #fff !important;                      /* FORCE white text */
#         text-shadow: 0 2.5px 14px rgba(15,20,36,0.38); /* Even stronger shadow */
#         font-weight: 900;
#         outline: 2.5px solid #fff;
#         box-shadow: 0 10px 28px 0 rgba(40,215,144,0.19);
#         filter: brightness(1.08) contrast(1.13);
#         transform: scale(1.05);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- File Loading ---
# @st.cache_resource
# def load_assets():
#     """Loads all necessary pre-trained components for making predictions."""
#     models_dir = 'models'
#     try:
#         model = joblib.load(os.path.join(models_dir, 'salary_prediction_model.pkl'))
#         scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
#         label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
#         training_columns = joblib.load(os.path.join(models_dir, 'training_columns.pkl'))
#         st.sidebar.success("Model and components loaded!")
#         return model, scaler, label_encoders, training_columns
#     except FileNotFoundError:
#         st.sidebar.error("Error: Model assets not found. Please run the 'Train and Save Model Assets' cell in your notebook first.")
#         return None, None, None, None

# model, scaler, label_encoders, training_columns = load_assets()

# # --- Main App Interface ---
# st.title("Employee Income Prediction System 🧑‍💻")
# st.markdown("---")

# if model:
#     st.markdown("<div style='text-align: center;'>This app uses a Machine Learning model to predict if an employee's income is >50K or <=50K.</div>", unsafe_allow_html=True)

#     # --- Single Prediction in Sidebar ---
#     st.sidebar.header("Single Prediction")
#     st.sidebar.subheader("Employee Details")

#     def get_options_from_encoder(encoder_name):
#         encoder = label_encoders.get(encoder_name)
#         return encoder.classes_.tolist() if encoder and hasattr(encoder, 'classes_') else []

#     # Input widgets
#     age = st.sidebar.slider("Age", 17, 90, 30)
#     workclass = st.sidebar.selectbox("Workclass", get_options_from_encoder('workclass'))
#     fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000)
#     educational_num = st.sidebar.slider("Education Years", 1, 16, 9)
#     marital_status = st.sidebar.selectbox("Marital Status", get_options_from_encoder('marital-status'))
#     occupation = st.sidebar.selectbox("Occupation", get_options_from_encoder('occupation'))
#     relationship = st.sidebar.selectbox("Relationship", get_options_from_encoder('relationship'))
#     race = st.sidebar.selectbox("Race", get_options_from_encoder('race'))
#     gender = st.sidebar.selectbox("Gender", get_options_from_encoder('gender'))
#     capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
#     capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
#     hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
#     native_country = st.sidebar.selectbox("Native Country", get_options_from_encoder('native-country'))

#     if st.sidebar.button("Predict Income 🚀"):
#         user_input_df = pd.DataFrame([{'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education-num': educational_num, 'marital-status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss, 'hours-per-week': hours_per_week, 'native-country': native_country}])

#         # One-hot encode the user input to match the training data structure
#         user_input_encoded = pd.get_dummies(user_input_df).reindex(columns=training_columns, fill_value=0)

#         prediction = model.predict(user_input_encoded)
#         prediction_proba = model.predict_proba(user_input_encoded)

#         st.subheader("Single Prediction Result")
#         predicted_label = label_encoders['income'].inverse_transform(prediction)[0]

#         if predicted_label == '>50K':
#             st.success(f"**Predicted Income: {predicted_label}** 🎉 (Confidence: {prediction_proba[0][1]*100:.2f}%)")
#         else:
#             st.warning(f"**Predicted Income: {predicted_label}** 😔 (Confidence: {prediction_proba[0][0]*100:.2f}%)")

#     # --- Batch Prediction on Main Page ---
#     st.markdown("---")
#     st.header("📂 Batch Prediction")
#     st.write("Upload a CSV file for batch income prediction.")

#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         batch_data_raw = pd.read_csv(uploaded_file)
#         st.write("Uploaded data preview:")
#         st.dataframe(batch_data_raw.head())

#         batch_data = batch_data_raw.copy()
#         if 'education' in batch_data.columns:
#             batch_data = batch_data.drop('education', axis=1)

#         batch_data.replace('?', np.nan, inplace=True)
#         for col in ['workclass', 'occupation', 'native-country']:
#             if col in batch_data.columns:
#                 batch_data[col].fillna(batch_data[col].mode()[0], inplace=True)

#         batch_encoded = pd.get_dummies(batch_data).reindex(columns=training_columns, fill_value=0)

#         with st.spinner('Predicting incomes for the batch...'):
#             batch_preds = model.predict(batch_encoded)
#             batch_proba = model.predict_proba(batch_encoded)

#         predicted_labels = label_encoders['income'].inverse_transform(batch_preds)
#         batch_data_raw['Predicted_Income'] = predicted_labels
#         batch_data_raw['Confidence'] = [f"{prob.max()*100:.2f}%" for prob in batch_proba]

#         st.success("✅ Batch predictions complete!")

#         # --- ADD THESE LINES ---
#         st.subheader("Prediction Results")
#         st.info("💡 The original data is shown below with two new columns added on the far right:")
#         st.markdown("""
#         * **Predicted_Income**: The model's prediction (`>50K` or `<=50K`).
#         * **Confidence**: The model's confidence in that prediction.
#         """)
#         # ----------------------

#         # 1. Define a function to style the columns
# def highlight_prediction_columns(s):
#     """
#     Applies bold, colored styling to the prediction columns.
#     You can change 'green' to any color you like.
#     """
#     color = 'green' 
#     return [f'color: {color}; font-weight: bold' for v in s]

# # 2. Apply the styling to your results dataframe for the specific columns
# styled_results = batch_data_raw.style.apply(
#     highlight_prediction_columns, 
#     subset=['Predicted_Income', 'Confidence']
# )

# # 3. Display the STYLED dataframe instead of the original one
# st.dataframe(styled_results)

#         csv = batch_data_raw.to_csv(index=False).encode('utf-8')
#         st.download_button(label="Download Predictions CSV", data=csv, file_name='predicted_incomes.csv', mime='text/csv')

# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os
# from sklearn.preprocessing import LabelEncoder

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Income Predictor",
#     page_icon="🧑‍💻",
#     layout="wide"
# )

# # --- UI Styling (Enhanced CSS) ---
# st.markdown(
#     """
#     <style>
#     /* General Styling */
#     .reportview-container, .main {
#         background: #f0f2f6;
#     }
#     /* Title Styling - Hides the anchor link icon */
#     h1 {
#         color: #007BFF;
#         text-align: center;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         font-weight: bold;
#     }
#     h1 a {
#         display: none !important;
#     }
#     h2, h3 { color: #0056b3; }
#     .stButton>button {
#         background-color: #28A745;
#         color: white;
#         border-radius: 8px;
#         padding: 10px 20px;
#         font-size: 16px;
#         font-weight: bold;
#         border: none;
#         box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
#         transition: background-color 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #218838;
#         box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
#     }
#     .sidebar .sidebar-content { background-color: #f8f9fa; }

#     [data-testid="stSidebar"] .stButton > button {
#         background: linear-gradient(90deg, #228f56 0%, #28d790 100%);
#         color: #fff !important;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         font-size: 18px;
#         font-weight: 800;
#         letter-spacing: 0.025em;
#         border: none;
#         border-radius: 30px;
#         text-shadow: 0 2px 13px rgba(20,32,34,0.33);
#         box-shadow: 0 3px 20px 0 rgba(34,143,86,0.14);
#         outline: 2px solid #28a745;
#         outline-offset: 1px;
#         padding: 14px 38px;
#         margin-top: 18px;
#         transition:
#             background 0.18s,
#             box-shadow 0.19s,
#             outline-color 0.13s,
#             transform 0.12s,
#             color 0.12s;
#     }
#     [data-testid="stSidebar"] .stButton > button:hover {
#         background: linear-gradient(90deg, #158349 0%, #27c982 100%);
#         color: #fff !important;                      /* FORCE white text */
#         text-shadow: 0 2.5px 14px rgba(15,20,36,0.38); /* Even stronger shadow */
#         font-weight: 900;
#         outline: 2.5px solid #fff;
#         box-shadow: 0 10px 28px 0 rgba(40,215,144,0.19);
#         filter: brightness(1.08) contrast(1.13);
#         transform: scale(1.05);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- File Loading ---
# @st.cache_resource
# def load_assets():
#     """Loads all necessary pre-trained components for making predictions."""
#     models_dir = 'models'
#     try:
#         model = joblib.load(os.path.join(models_dir, 'salary_prediction_model.pkl'))
#         scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
#         label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
#         training_columns = joblib.load(os.path.join(models_dir, 'training_columns.pkl'))
#         st.sidebar.success("Model and components loaded!")
#         return model, scaler, label_encoders, training_columns
#     except FileNotFoundError:
#         st.sidebar.error("Error: Model assets not found. Please ensure the 'models' folder is in the same directory as app.py.")
#         return None, None, None, None

# model, scaler, label_encoders, training_columns = load_assets()

# # --- Main App Interface ---
# st.title("Employee Income Prediction System 🧑‍💻")
# st.markdown("---")

# if model:
#     st.markdown("<div style='text-align: center;'>This app uses a Machine Learning model to predict if an employee's income is >50K or <=50K.</div>", unsafe_allow_html=True)

#     # --- Single Prediction in Sidebar ---
#     st.sidebar.header("Single Prediction")
#     st.sidebar.subheader("Employee Details")

#     def get_options_from_encoder(encoder_name):
#         encoder = label_encoders.get(encoder_name)
#         return encoder.classes_.tolist() if encoder and hasattr(encoder, 'classes_') else []

#     # Input widgets
#     age = st.sidebar.slider("Age", 17, 90, 30)
#     workclass = st.sidebar.selectbox("Workclass", get_options_from_encoder('workclass'))
#     fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000)
#     educational_num = st.sidebar.slider("Education Years", 1, 16, 9)
#     marital_status = st.sidebar.selectbox("Marital Status", get_options_from_encoder('marital-status'))
#     occupation = st.sidebar.selectbox("Occupation", get_options_from_encoder('occupation'))
#     relationship = st.sidebar.selectbox("Relationship", get_options_from_encoder('relationship'))
#     race = st.sidebar.selectbox("Race", get_options_from_encoder('race'))
#     gender = st.sidebar.selectbox("Gender", get_options_from_encoder('gender'))
#     capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
#     capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
#     hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
#     native_country = st.sidebar.selectbox("Native Country", get_options_from_encoder('native-country'))

#     if st.sidebar.button("Predict Income 🚀"):
#         user_input_df = pd.DataFrame([{'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education-num': educational_num, 'marital-status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss, 'hours-per-week': hours_per_week, 'native-country': native_country}])
#         user_input_encoded = pd.get_dummies(user_input_df).reindex(columns=training_columns, fill_value=0)

#         prediction = model.predict(user_input_encoded)
#         prediction_proba = model.predict_proba(user_input_encoded)

#         st.subheader("Single Prediction Result")
#         predicted_label = label_encoders['income'].inverse_transform(prediction)[0]

#         if predicted_label == '>50K':
#             st.success(f"**Predicted Income: {predicted_label}** 🎉 (Confidence: {prediction_proba[0][1]*100:.2f}%)")
#         else:
#             st.warning(f"**Predicted Income: {predicted_label}** 😔 (Confidence: {prediction_proba[0][0]*100:.2f}%)")

#     # --- Batch Prediction on Main Page ---
#     st.markdown("---")
#     st.header("📂 Batch Prediction")
#     st.write("Upload a CSV file for batch income prediction.")

#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         batch_data_raw = pd.read_csv(uploaded_file)
#         st.write("Uploaded data preview:")
#         st.dataframe(batch_data_raw.head())

#         # --- PREDICTION LOGIC (on the full file) ---
#         batch_data = batch_data_raw.copy()
#         if 'education' in batch_data.columns:
#             batch_data = batch_data.drop('education', axis=1)

#         batch_data.replace('?', np.nan, inplace=True)
#         for col in ['workclass', 'occupation', 'native-country']:
#             if col in batch_data.columns:
#                 batch_data[col].fillna(batch_data[col].mode()[0], inplace=True)

#         batch_encoded = pd.get_dummies(batch_data).reindex(columns=training_columns, fill_value=0)

#         with st.spinner('Predicting incomes for the batch...'):
#             batch_preds = model.predict(batch_encoded)
#             batch_proba = model.predict_proba(batch_encoded)

#         # Add predictions to the original full dataframe
#         predicted_labels = label_encoders['income'].inverse_transform(batch_preds)
#         batch_data_raw['Predicted_Income'] = predicted_labels
#         batch_data_raw['Confidence'] = [f"{prob.max()*100:.2f}%" for prob in batch_proba]

#         st.success("✅ Batch predictions complete!")

#         # --- UX MESSAGES AND STYLED PREVIEW ---
#         st.subheader("Prediction Results")
#         st.info("💡 Displaying a preview of the first 100 results with highlighted predictions. Use the download button for the full file.")
#         st.markdown("""
#         * **Predicted_Income**: The model's prediction (`>50K` or `<=50K`).
#         * **Confidence**: The model's confidence in that prediction.
#         """)

#         # 1. Create a smaller preview dataframe
#         preview_df = batch_data_raw.head(100)

#         # 2. Define the styling function
#         def highlight_prediction_columns(s):
#             color = 'green'
#             return [f'color: {color}; font-weight: bold' for v in s]

#         # 3. Apply styling ONLY to the small preview dataframe
#         styled_preview = preview_df.style.apply(
#             highlight_prediction_columns,
#             subset=['Predicted_Income', 'Confidence']
#         )

#         # 4. Display the fast, styled preview
#         st.dataframe(styled_preview)

#         # --- DOWNLOAD BUTTON (for the full file) ---
#         csv = batch_data_raw.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="Download Full Predictions CSV",
#             data=csv,
#             file_name='predicted_incomes.csv',
#             mime='text/csv'
#         )

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Income Predictor",
    page_icon="🧑‍💻",
    layout="wide"
)

# --- UI Styling (Enhanced CSS) ---
st.markdown(
    """
    <style>
    /* General Styling */
    .reportview-container, .main {
        background: #f0f2f6;
    }
    /* Title Styling - Hides the anchor link icon */
    h1 {
        color: #007BFF;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: bold;
    }
    h1 a {
        display: none !important;
    }
    h2, h3 { color: #0056b3; }
    .stButton>button {
        background-color: #28A745;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    .sidebar .sidebar-content { background-color: #f8f9fa; }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(90deg, #228f56 0%, #28d790 100%);
        color: #fff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.025em;
        border: none;
        border-radius: 30px;
        text-shadow: 0 2px 13px rgba(20,32,34,0.33);
        box-shadow: 0 3px 20px 0 rgba(34,143,86,0.14);
        outline: 2px solid #28a745;
        outline-offset: 1px;
        padding: 14px 38px;
        margin-top: 18px;
        transition:
            background 0.18s,
            box-shadow 0.19s,
            outline-color 0.13s,
            transform 0.12s,
            color 0.12s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(90deg, #158349 0%, #27c982 100%);
        color: #fff !important;                      /* FORCE white text */
        text-shadow: 0 2.5px 14px rgba(15,20,36,0.38); /* Even stronger shadow */
        font-weight: 900;
        outline: 2.5px solid #fff;
        box-shadow: 0 10px 28px 0 rgba(40,215,144,0.19);
        filter: brightness(1.08) contrast(1.13);
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- File Loading ---
@st.cache_resource
def load_assets():
    """Loads all necessary pre-trained components for making predictions."""
    models_dir = 'models'
    try:
        model = joblib.load(os.path.join(models_dir, 'salary_prediction_model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        training_columns = joblib.load(os.path.join(models_dir, 'training_columns.pkl'))
        st.sidebar.success("Model and components loaded!")
        return model, scaler, label_encoders, training_columns
    except FileNotFoundError:
        st.sidebar.error("Error: Model assets not found. Please ensure the 'models' folder is in the same directory as app.py.")
        return None, None, None, None

model, scaler, label_encoders, training_columns = load_assets()

# --- Main App Interface ---
st.title("Employee Income Prediction System 🧑‍�")
st.markdown("---")

if model:
    st.markdown("<div style='text-align: center;'>This app uses a Machine Learning model to predict if an employee's income is >50K or <=50K.</div>", unsafe_allow_html=True)

    # --- Single Prediction in Sidebar ---
    st.sidebar.header("Single Prediction")
    st.sidebar.subheader("Employee Details")

    def get_options_from_encoder(encoder_name):
        encoder = label_encoders.get(encoder_name)
        return encoder.classes_.tolist() if encoder and hasattr(encoder, 'classes_') else []

    # Input widgets
    age = st.sidebar.slider("Age", 17, 90, 30)
    workclass = st.sidebar.selectbox("Workclass", get_options_from_encoder('workclass'))
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000)
    educational_num = st.sidebar.slider("Education Years", 1, 16, 9)
    marital_status = st.sidebar.selectbox("Marital Status", get_options_from_encoder('marital-status'))
    occupation = st.sidebar.selectbox("Occupation", get_options_from_encoder('occupation'))
    relationship = st.sidebar.selectbox("Relationship", get_options_from_encoder('relationship'))
    race = st.sidebar.selectbox("Race", get_options_from_encoder('race'))
    gender = st.sidebar.selectbox("Gender", get_options_from_encoder('gender'))
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", get_options_from_encoder('native-country'))

    if st.sidebar.button("Predict Income 🚀"):
        user_input_df = pd.DataFrame([{'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education-num': educational_num, 'marital-status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss, 'hours-per-week': hours_per_week, 'native-country': native_country}])
        user_input_encoded = pd.get_dummies(user_input_df).reindex(columns=training_columns, fill_value=0)

        prediction = model.predict(user_input_encoded)
        prediction_proba = model.predict_proba(user_input_encoded)

        st.subheader("Single Prediction Result")
        predicted_label = label_encoders['income'].inverse_transform(prediction)[0]

        if predicted_label == '>50K':
            st.success(f"**Predicted Income: {predicted_label}** 🎉 (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        else:
            st.warning(f"**Predicted Income: {predicted_label}** 😔 (Confidence: {prediction_proba[0][0]*100:.2f}%)")

    # --- Batch Prediction on Main Page ---
    st.markdown("---")
    st.header("📂 Batch Prediction")
    st.write("Upload a CSV file for batch income prediction.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        batch_data_raw = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(batch_data_raw.head())

        # --- PREDICTION LOGIC (on the full file) ---
        batch_data = batch_data_raw.copy()
        if 'education' in batch_data.columns:
            batch_data = batch_data.drop('education', axis=1)

        batch_data.replace('?', np.nan, inplace=True)
        for col in ['workclass', 'occupation', 'native-country']:
            if col in batch_data.columns:
                batch_data[col].fillna(batch_data[col].mode()[0], inplace=True)

        batch_encoded = pd.get_dummies(batch_data).reindex(columns=training_columns, fill_value=0)

        with st.spinner('Predicting incomes for the batch...'):
            batch_preds = model.predict(batch_encoded)
            batch_proba = model.predict_proba(batch_encoded)

        # Add predictions to the original full dataframe
        predicted_labels = label_encoders['income'].inverse_transform(batch_preds)
        batch_data_raw['Predicted_Income'] = predicted_labels
        batch_data_raw['Confidence'] = [f"{prob.max()*100:.2f}%" for prob in batch_proba]

        st.success("✅ Batch predictions complete!")

        # --- UX MESSAGES ---
        st.subheader("Prediction Results")
        st.info("💡 Browse through the results using the page selector below. Use the download button for the full file.")
        st.markdown("""
        * **Predicted_Income**: The model's prediction (`>50K` or `<=50K`).
        * **Confidence**: The model's confidence in that prediction.
        """)

        # --- PAGINATION AND STYLING LOGIC ---
        page_size = 100
        total_rows = len(batch_data_raw)
        total_pages = (total_rows // page_size) + (1 if total_rows % page_size > 0 else 0)

        page_number = st.number_input('Select Page', min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        
        # Slice the dataframe for the current page
        page_df = batch_data_raw.iloc[start_idx:end_idx]

        # Define the styling function
        def highlight_prediction_columns(s):
            color = 'green'
            return [f'color: {color}; font-weight: bold' for v in s]

        # Apply styling ONLY to the small page dataframe
        styled_page = page_df.style.apply(
            highlight_prediction_columns,
            subset=['Predicted_Income', 'Confidence']
        )

        # Display the fast, styled page
        st.write(f"Showing rows {start_idx+1} to {min(end_idx, total_rows)} of {total_rows}")
        st.dataframe(styled_page)

        # --- DOWNLOAD BUTTON (for the full file) ---
        csv = batch_data_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Predictions CSV",
            data=csv,
            file_name='predicted_incomes.csv',
            mime='text/csv'
        )