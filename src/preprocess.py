# STEP 1: Run this script first to create federated_data/

import os
import numpy as np
import src.config as config
from src.utils import get_subject_labels, get_clean_signals

def create_windows(signals, labels, window_len, window_step):
    """
    Creates overlapping windows from the signal data.
    Only keeps windows that are *fully* 'baseline' (1) or 'stress' (2).
    """
    X_windows = []
    
    num_channels, total_len = signals.shape
    
    for start in range(0, total_len - window_len + 1, window_step):
        end = start + window_len
        
        window_labels = labels[start:end]
        
        # We only want windows from baseline (1) or stress (2)
        # Check the mode (most frequent label) in the window
        window_label_mode = np.bincount(window_labels).argmax()
        
        if window_label_mode == 1 or window_label_mode == 2:
            X_windows.append(signals[:, start:end])
            
    if not X_windows:
        return None
            
    return np.array(X_windows)

def main():
    """
    Main preprocessing script.
    Loops through all subjects in RAW_WESAD_DIR.
    Saves their processed (X_windows, y_trait_labels) to FED_DATA_DIR.
    """
    os.makedirs(config.FED_DATA_DIR, exist_ok=True)
    subjects = [d for d in os.listdir(config.RAW_WESAD_DIR) if d.startswith("S")]
    
    for subject_id in subjects:
        print(f"--- Processing {subject_id} ---")
        subject_dir_path = os.path.join(config.RAW_WESAD_DIR, subject_id)
        
        # 1. Get Subject-level trait labels
        quest_path = os.path.join(subject_dir_path, f"{subject_id}_quest.csv")
        subject_trait_labels = get_subject_labels(quest_path)
        
        if np.isnan(subject_trait_labels).any():
            print(f"      -> Skipping {subject_id}: Could not parse labels.")
            continue
            
        # 2. Get Clean Signals
        pkl_path = os.path.join(subject_dir_path, f"{subject_id}.pkl")
        clean_signals, clean_labels = get_clean_signals(pkl_path, config.TARGET_SR)
        
        # 3. Create Windows
        X_windows = create_windows(
            clean_signals, clean_labels, config.WINDOW_LEN, config.WINDOW_STEP
        )
        
        if X_windows is None:
            print(f"      -> Skipping {subject_id}: No valid windows found.")
            continue
            
        # 4. Create the y labels
        # Pair EVERY window with the SAME two trait labels.
        y_trait_labels = np.tile(subject_trait_labels, (len(X_windows), 1))

        # 5. Save to this client's federated data folder
        client_dir = os.path.join(config.FED_DATA_DIR, subject_id)
        os.makedirs(client_dir, exist_ok=True)
        
        save_path = os.path.join(client_dir, "data.npz")
        np.savez(save_path, X=X_windows, y=y_trait_labels)
        
        print(f"✅ Saved data for {subject_id} to {save_path}")
        print(f"   X shape: {X_windows.shape}")
        print(f"   y shape: {y_trait_labels.shape}")

if __name__ == "__main__":
    main()