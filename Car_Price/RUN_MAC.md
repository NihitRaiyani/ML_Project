# 🚗 Running Car Price Prediction App on Mac

## ✅ Quick Start (Recommended)

### Method 1: Use the Run Script (Easiest)

Open **Terminal** on your Mac and run:

```bash
cd "/Users/nihitraiyani/ML_project/Car Price Prediction"
./run_app.sh
```

This will automatically:
- Activate the virtual environment
- Install dependencies if needed
- Start the app

---

## 📝 Manual Method (Step by Step)

### Step 1: Open Terminal
- Press `⌘ (Command) + Space` to open Spotlight
- Type "Terminal" and press Enter
- OR go to Applications → Utilities → Terminal

### Step 2: Navigate to Project Directory
```bash
cd "/Users/nihitraiyani/ML_project"
```

### Step 3: Activate Virtual Environment
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Go to App Directory
```bash
cd "Car Price Prediction"
```

### Step 5: Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### Step 6: Run the App
```bash
streamlit run app.py
```

---

## 🎯 What Happens Next

1. **Terminal will show:**
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. **Your browser will automatically open** to `http://localhost:8501`

3. **If it doesn't open automatically**, manually open Safari/Chrome and go to:
   ```
   http://localhost:8501
   ```

---

## 🛑 To Stop the App

Press `Ctrl + C` in the Terminal window

---

## ⚠️ Troubleshooting for Mac

### Issue 1: Permission Denied when running script
**Solution:**
```bash
chmod +x run_app.sh
./run_app.sh
```

### Issue 2: Python/Streamlit not found
**Solution:** Install using Homebrew:
```bash
brew install python3
pip3 install streamlit pandas numpy scikit-learn xgboost lightgbm joblib plotly
```

### Issue 3: Port 8501 already in use
**Solution:** Run on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue 4: ModuleNotFoundError
**Solution:** Make sure virtual environment is activated:
```bash
source ../venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quick One-Liner (Copy & Paste)

```bash
cd "/Users/nihitraiyani/ML_project" && source venv/bin/activate && cd "Car Price Prediction" && streamlit run app.py
```

---

## 📱 Using the App

Once running, you'll see:

- **🏠 Home**: Overview and statistics
- **🤖 Train Model**: Train ML models (takes a few minutes)
- **💰 Price Predictor**: Predict car prices
- **📊 Data Analysis**: Interactive visualizations
- **📈 Model Info**: View model details

Enjoy your Car Price Prediction App! 🎉

