# How to Run Car Price Prediction App

## Step 1: Navigate to the Project Directory
```bash
cd "/Users/nihitraiyani/ML_project/Car Price Prediction"
```

## Step 2: Activate Virtual Environment (if you have one)
If you're using a virtual environment:
```bash
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows
```

## Step 3: Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm joblib plotly
```

## Step 4: Run the Streamlit App
```bash
streamlit run app.py
```

## What to Expect:

1. The terminal will show:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. Your browser will automatically open to `http://localhost:8501`

3. If it doesn't open automatically, manually open your browser and go to:
   ```
   http://localhost:8501
   ```

## Using the App:

1. **🏠 Home**: View dataset overview and statistics
2. **🤖 Train Model**: Train machine learning models (this takes a few minutes)
3. **💰 Price Predictor**: Predict car prices using trained model
4. **📊 Data Analysis**: Explore data with interactive visualizations
5. **📈 Model Info**: View model details and information

## Troubleshooting:

### Issue: Module not found error
**Solution**: Install missing packages
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm joblib plotly
```

### Issue: Port 8501 already in use
**Solution**: Run on a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Data file not found
**Solution**: Make sure `Cleaned_Car_data.csv` is in the same directory as `app.py`

### Issue: Model file not found
**Solution**: Train a model first by going to "🤖 Train Model" page in the app

## Stop the App:
Press `Ctrl + C` in the terminal to stop the Streamlit server

