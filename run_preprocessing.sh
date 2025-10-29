#!/bin/bash
# run_preprocessing.sh
# This script processes the raw WESAD data into the federated_data/ directory.
# Run this script ONCE before starting the FL simulation.

echo "--- Starting WESAD Preprocessing ---"

# Ensure we are running from the project root
if [ ! -d "src" ]; then
    echo "Error: This script must be run from the project root directory (wesad-fl-project/)"
    exit 1
fi

# Run the preprocessing script
python -u src/preprocess.py

echo "--- Preprocessing Complete ---"
echo "Check the 'federated_data/' directory for client data."