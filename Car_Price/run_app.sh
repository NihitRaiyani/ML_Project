#!/bin/bash

# Car Price Prediction App - Quick Start Script

echo "🚗 Starting Car Price Prediction App..."
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists in parent directory
if [ -d "../venv" ]; then
    echo "📦 Activating virtual environment from ../venv..."
    source ../venv/bin/activate
    echo "✅ Virtual environment activated!"
    echo ""
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if data file exists
if [ ! -f "Cleaned_Car_data.csv" ]; then
    echo "❌ Error: Cleaned_Car_data.csv not found!"
    echo "Please make sure the data file is in the same directory."
    exit 1
fi

echo "✅ All checks passed!"
echo ""
echo "🚀 Starting Streamlit app..."
echo "   The app will open automatically in your browser at http://localhost:8501"
echo ""
echo "   To stop the app, press Ctrl+C"
echo ""

# Run the app
streamlit run app.py

