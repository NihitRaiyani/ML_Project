#!/bin/bash

# Quick Start Script for Advanced Car Price Prediction System

echo "=========================================="
echo "🚗 Advanced Car Price Prediction System"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your internet connection."
    exit 1
fi

echo "✅ Dependencies installed successfully!"
echo ""

# Check if model exists
if [ ! -f "best_car_price_model.pkl" ]; then
    echo "📊 Training models (this may take 5-10 minutes)..."
    python3 model_trainer.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Model training failed. Please check the error messages above."
        exit 1
    fi
    
    echo "✅ Models trained successfully!"
else
    echo "✅ Model file found. Skipping training."
    echo "   (To retrain, delete best_car_price_model.pkl and run this script again)"
fi

echo ""
echo "=========================================="
echo "🚀 Starting Web Application..."
echo "=========================================="
echo ""
echo "The application will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit app
streamlit run app.py

