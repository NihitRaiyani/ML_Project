# Medical Insurance Cost Predictor

A Streamlit web application for predicting medical insurance costs based on patient information.

## Files
- `insurance.csv` - Dataset with medical insurance information
- `medical_insurance_model.pkl` - Trained machine learning model
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `run.sh` - Quick start script (Mac/Linux)

## Quick Start (Easiest Way)

### Option 1: Using the run script (Mac/Linux)
```bash
./run.sh
```

### Option 2: Manual setup
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

**IMPORTANT**: Always use `streamlit run app.py` NOT `python app.py`

The app will open in your default web browser at `http://localhost:8501`

## Features

- **Insurance Cost Prediction**: Enter patient information (age, sex, BMI, children, smoker status, region) to predict insurance costs
- **Dataset Overview**: View statistics and preview of the insurance dataset
- **Interactive Interface**: User-friendly sidebar for input and main area for results

## Usage

1. Enter patient information in the sidebar
2. Click "Predict Insurance Cost" button
3. View the predicted cost and input summary
4. Explore the dataset statistics in the main area

## Troubleshooting

- If you see "File not found" errors, make sure you're running the app from the directory containing `app.py`, `insurance.csv`, and `medical_insurance_model.pkl`
- If you see "missing ScriptRunContext" warnings, you're running with `python` instead of `streamlit run` - use `streamlit run app.py` instead
