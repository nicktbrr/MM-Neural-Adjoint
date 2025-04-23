#!/bin/bash

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install the package in development mode
pip install -e .

echo "Package installed successfully!" 