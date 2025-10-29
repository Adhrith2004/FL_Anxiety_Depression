# 💡 All hyperparameters and paths

# --- 1. File Paths ---
RAW_WESAD_DIR = "data/WESAD"
# Where the preprocessed client data will be saved
FED_DATA_DIR = "federated_data"

# --- 2. Preprocessing Hyperparameters ---
# WESAD signals are 700 Hz. We'll downsample to 100 Hz.
ORIGINAL_SR = 700
TARGET_SR = 100

# We'll use 10-second windows with 50% overlap
WINDOW_SEC = 10
WINDOW_OVERLAP = 0.5 

WINDOW_LEN = int(TARGET_SR * WINDOW_SEC) # 1000 samples
WINDOW_STEP = int(WINDOW_LEN * (1.0 - WINDOW_OVERLAP)) # 500 samples

# --- 3. FL Simulation Parameters ---
# Address for the Flower server
SERVER_ADDRESS = "127.0.0.1:8080"
# Number of global training rounds
NUM_ROUNDS = 3
# Number of local epochs each client trains for
LOCAL_EPOCHS = 2
# Batch size for local training
BATCH_SIZE = 32