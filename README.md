# Federated Multi-Task Learning for Anxiety and Negative Affect Prediction

This is a Federated Learning (FL) system to predict mental health traits from physiological signals. The model is trained on the WESAD dataset.

## Project Goal

The system trains a single **1D-CNN + LSTM** model to perform **Multi-Task Learning**. It takes 10-second windows of raw physiological data (ECG, EDA, BVP) as input and jointly predicts two separate outputs:

1.  **Anxiety Tendency:** (Regression) Predicts the subject's **STAI-T score**.
2.  **Depression Tendency:** (Regression) Predicts the subject's **PANAS-N score** (Negative Affect), which serves as a validated proxy for depressive mood.

The model is trained using a federated simulation (Flower) where each subject (e.g., "S2") acts as a client, ensuring their private physiological data is never shared.

## How to Run

### 1. Setup

1.  Clone this repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add WESAD Data:** Download the WESAD dataset. Place the subject folders (e.g., `S2`, `S3`...) inside the `data/WESAD/` directory.

### 2. Step 1: Run Preprocessing

You must run this script **once** to process the raw WESAD data into a federated structure. It will populate the `federated_data/` directory.

```bash
bash run_preprocessing.sh
```

### 3. Step 2: Run the FL Simulation
This script starts the FL server and multiple clients in parallel to simulate the training process. The server will coordinate the training for the number of rounds specified in src/config.py.

```bash

bash run_simulation.sh
```
This will start a server and two clients (S2, S3) by default.

To run with more clients, edit the CLIENTS variable in run_simulation.sh.