# Signal processing helper functions

import numpy as np
import pickle
from scipy.signal import butter, filtfilt, resample
from src.config import ORIGINAL_SR

def get_subject_labels(quest_csv_path):
    """
    Parses the _quest.csv file to find the STAI-T (Anxiety)
    and PANAS-N (Negative Affect) scores.
    
    Returns:
        np.array: [stai_score, panas_n_score]
    """
    stai_score = np.nan
    panas_n_score = np.nan
    
    try:
        with open(quest_csv_path, 'r') as f:
            for line in f:
                if 'STAI-T;' in line:
                    stai_score = float(line.split(';')[1].strip())
                elif 'PANAS-N;' in line:
                    panas_n_score = float(line.split(';')[1].strip())
                    
        if np.isnan(stai_score) or np.isnan(panas_n_score):
            raise ValueError("Could not find STAI or PANAS scores")
            
        return np.array([stai_score, panas_n_score], dtype=np.float32)
    
    except Exception as e:
        print(f"Error parsing {quest_csv_path}: {e}")
        return np.array([np.nan, np.nan], dtype=np.float32)

def get_clean_signals(pkl_path, target_sr):
    """
    Loads pkl file, extracts, cleans (filters), and resamples signals.
    We select 3 key signals:
    1. ECG (Chest)
    2. EDA (Chest)
    3. BVP (Wrist) - Blood Volume Pulse
    """
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    signals = data['signal']
    labels = data['label'].flatten()

    ecg_raw = signals['chest']['ECG'].flatten()
    eda_raw = signals['chest']['EDA'].flatten()
    bvp_raw = signals['wrist']['BVP'].flatten()
    
    # --- Filter *before* resampling to prevent aliasing ---
    b, a = butter(3, [0.5, 40], btype='bandpass', fs=ORIGINAL_SR)
    ecg_filt = filtfilt(b, a, ecg_raw)
    
    b, a = butter(3, 1, btype='low', fs=ORIGINAL_SR)
    eda_filt = filtfilt(b, a, eda_raw)
    
    b, a = butter(3, [0.5, 8], btype='bandpass', fs=64) # BVP is 64Hz
    bvp_filt = filtfilt(b, a, bvp_raw)

    # --- Resample to Target SR ---
    ecg_resampled = resample(ecg_filt, int(len(ecg_filt) * (target_sr / ORIGINAL_SR)))
    eda_resampled = resample(eda_filt, int(len(eda_filt) * (target_sr / ORIGINAL_SR)))
    bvp_resampled = resample(bvp_filt, int(len(bvp_filt) * (target_sr / 64)))
    
    # --- Synchronize (truncate to shortest) ---
    min_len = min(len(ecg_resampled), len(eda_resampled), len(bvp_resampled))
    
    labels_resampled = resample(labels, min_len, window='boxcar') 
    labels_resampled = np.round(labels_resampled).astype(int)

    final_signals = np.stack([
        ecg_resampled[:min_len],
        eda_resampled[:min_len],
        bvp_resampled[:min_len]
    ], axis=0) # Shape: (3, min_len)
    
    # Z-score normalization per channel
    final_signals = (final_signals - np.mean(final_signals, axis=1, keepdims=True)) \
                    / (np.std(final_signals, axis=1, keepdims=True) + 1e-6)

    return final_signals, labels_resampled