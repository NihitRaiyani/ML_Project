# 🚗 Advanced Car Price Prediction System

An advanced-level machine learning system for predicting car prices using multiple ML models with feature engineering, hyperparameter tuning, and an interactive web interface.

## ✨ Features

- **Multi-Model ML System**: Implements 4 advanced ML algorithms
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - LightGBM Regressor
- **Hyperparameter Tuning**: Automatic optimization using GridSearchCV
- **Advanced Feature Engineering**: 
  - Car age calculation
  - Kilometers per year (usage intensity)
  - Brand extraction and encoding
  - Categorical encoding
- **Interactive Web Interface**: Beautiful Streamlit app with:
  - Real-time price predictions
  - Interactive data visualizations
  - Price trends analysis
  - Company and fuel type comparisons
  - Model performance metrics
- **Model Selection**: Automatically selects the best performing model
- **Data Analysis**: Comprehensive analysis tools and visualizations

## 📋 Requirements

Python 3.8 or higher

## 🚀 Installation

1. Clone or navigate to this directory:
```bash
cd "Car Price Prediction"
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Train the Models

First, train the ML models to create the prediction system:

```bash
python model_trainer.py
```

This will:
- Load and preprocess the car data
- Perform feature engineering
- Train 4 different ML models with hyperparameter tuning
- Evaluate all models and select the best one
- Save the best model to `best_car_price_model.pkl`

**Note:** Training may take 5-10 minutes depending on your system.

### Step 2: Launch the Web Application

Start the Streamlit web app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🎯 Using the Web Application

### Home Page
- Overview of the dataset
- Quick statistics
- Top companies visualization

### Price Predictor
1. Select the car company
2. Choose the car model
3. Select fuel type (Petrol/Diesel/LPG)
4. Adjust the year using the slider
5. Enter kilometers driven
6. Click "Predict Price" to get an instant price estimate

### Data Analysis
- Interactive filters for companies, years, and price ranges
- **Price Trends**: See how prices change over years
- **By Company**: Compare prices across different car companies
- **By Fuel Type**: Analyze price differences between fuel types
- **Distributions**: View price distributions and correlations

### Model Performance
- View current model information
- See feature columns used
- Understand categorical encoders
- Instructions for retraining

## 📊 Dataset

The system uses `Cleaned_Car_data.csv` which contains:
- **name**: Car model name
- **company**: Car manufacturer
- **year**: Manufacturing year
- **Price**: Car price (target variable)
- **kms_driven**: Kilometers driven
- **fuel_type**: Type of fuel (Petrol/Diesel/LPG)

## 🧠 Model Details

### Feature Engineering

The system creates several derived features:
1. **car_age**: Current year - manufacturing year
2. **kms_per_year**: Kilometers driven per year (usage intensity)
3. **brand_encoded**: Encoded brand name
4. **company_encoded**: Encoded company name
5. **fuel_encoded**: Encoded fuel type

### Model Evaluation

Models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)
- **Cross-Validation**: 5-fold CV for robust evaluation

The best model is selected based on the highest R² score.

## 📁 Project Structure

```
Car Price Prediction/
├── app.py                    # Streamlit web application
├── model_trainer.py          # ML model training script
├── Cleaned_Car_data.csv      # Dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── best_car_price_model.pkl  # Saved model (created after training)
└── LinearRegressionModel.pkl # Old model (if exists)
```

## 🔧 Advanced Configuration

### Modify Model Parameters

Edit `model_trainer.py` to adjust:
- Hyperparameter search space
- Number of cross-validation folds
- Train/test split ratio
- Additional feature engineering

### Customize Web Interface

Edit `app.py` to:
- Add new visualizations
- Modify UI styling
- Add additional features

## 📈 Performance Tips

1. **Training**: Use a machine with multiple CPU cores for faster training
2. **Prediction**: Models are optimized for quick inference
3. **Data Updates**: Retrain models periodically with new data

## 🤝 Contributing

Feel free to enhance the system by:
- Adding more ML models
- Implementing deep learning models
- Adding more features
- Improving UI/UX

## 📝 License

This project is open source and available for educational purposes.

## ⚠️ Notes

- Ensure `Cleaned_Car_data.csv` is in the same directory
- Train models before running the web app
- Model training requires sufficient system resources
- Predictions are estimates based on historical data patterns

---

**Built with ❤️ using Python, Scikit-learn, XGBoost, LightGBM, and Streamlit**

