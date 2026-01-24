#!/bin/bash
# Script to run the Streamlit app

# Navigate to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py
