#!/bin/bash
# run_simulation.sh
# This script starts a federated learning simulation.
# It launches the server and two clients in parallel.

echo "--- Starting FL Simulation ---"

# This function will be called when the script exits (e.g., Ctrl+C)
trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM
# Kills all processes in the script's process group

# 1. Start the server in the background
echo "Starting server..."
python -u src/server.py &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
sleep 5 # Wait for the server to initialize

# 2. Start the clients
# Add/remove clients here.
# Make sure the 'cid' matches the folder names in 'federated_data/'
CLIENTS="S2 S3"

for cid in $CLIENTS; do
    echo "Starting client $cid..."
    python -u src/client.py --cid "$cid" &
done

# Wait for all background processes (server and clients) to finish
wait
echo "--- Simulation Complete ---"