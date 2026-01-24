import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥", layout="wide")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(script_dir, 'medical_insurance_model.pkl')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Load the CSV data
@st.cache_data
def load_data():
    try:
        csv_path = os.path.join(script_dir, 'insurance.csv')
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found at: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Main title
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("---")

# Load model and data
model = load_model()
df = load_data()

# Check if loading was successful
if model is None:
    st.error("❌ Failed to load model. Please check that 'medical_insurance_model.pkl' exists in the same directory.")
    st.stop()

if df is None:
    st.error("❌ Failed to load data. Please check that 'insurance.csv' exists in the same directory.")
    st.stop()

# Sidebar for user input
st.sidebar.header("📝 Enter Patient Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare input data for prediction
# Create input dataframe matching the CSV structure
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Function to preprocess input data to match model expectations
def preprocess_input(age, sex, bmi, children, smoker, region, model=None):
    """Preprocess input data to match the model's expected format"""
    # Create a dataframe with the input
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Encode categorical variables
    # Sex: male=1, female=0
    input_df['sex'] = input_df['sex'].map({'male': 1, 'female': 0})
    
    # Smoker: yes=1, no=0
    input_df['smoker'] = input_df['smoker'].map({'yes': 1, 'no': 0})
    
    # One-hot encode region - ensure all region columns are present
    region_columns = ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
    region_dummies = pd.get_dummies(input_df['region'], prefix='region')
    
    # Add missing region columns (set to 0)
    for col in region_columns:
        if col not in region_dummies.columns:
            region_dummies[col] = 0
    
    # Ensure columns are in the correct order
    region_dummies = region_dummies[region_columns]
    
    # Combine all features
    input_encoded = pd.concat([
        input_df[['age', 'sex', 'bmi', 'children', 'smoker']],
        region_dummies
    ], axis=1)
    
    # If model has feature_names_in_, reorder to match exactly
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in input_encoded.columns:
                input_encoded[feat] = 0
        # Reorder to match model's expected order
        input_encoded = input_encoded[expected_features]
    
    return input_encoded

# Prediction button
if st.sidebar.button("🔮 Predict Insurance Cost", type="primary"):
    try:
        # Check if model is a pipeline (might include preprocessing)
        from sklearn.pipeline import Pipeline
        
        # Make prediction
        if isinstance(model, Pipeline):
            # Pipeline should handle preprocessing automatically
            prediction = model.predict(input_data)[0]
        else:
            # For non-pipeline models, we need to preprocess the data
            # Preprocess input to match model's expected format
            input_encoded = preprocess_input(age, sex, bmi, children, smoker, region, model)
            
            # Make prediction
            prediction = model.predict(input_encoded)[0]
        
        # Display result
        st.success(f"### Predicted Insurance Cost: ${prediction:,.2f}")
        
        # Show input summary
        st.info(f"""
        **Input Summary:**
        - Age: {age}
        - Sex: {sex}
        - BMI: {bmi}
        - Children: {children}
        - Smoker: {smoker}
        - Region: {region}
        """)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Error details:", str(e))
        import traceback
        st.code(traceback.format_exc())
        
        # Debug information
        if hasattr(model, 'feature_names_in_'):
            st.write("**Model expects these features:**", list(model.feature_names_in_))

# Main content area - only show if data loaded successfully
if df is not None and not df.empty:
    st.header("📊 Dataset Overview")
    
    # Display basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        st.metric("Average Age", f"{df['age'].mean():.1f}")
    
    with col3:
        st.metric("Average BMI", f"{df['bmi'].mean():.2f}")
    
    with col4:
        st.metric("Average Charges", f"${df['charges'].mean():,.2f}")
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Display statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
