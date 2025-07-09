import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os

def extract_ppg(csv_path):
    df = pd.read_csv(csv_path)
    red_values = df.iloc[:, 1].astype(float)  # second column
    timestamps = df.iloc[:, 0].astype(float)

    fs = 30  # sampling rate (30 fps)

    # Bandpass filter
    def bandpass(data, lowcut=0.5, highcut=4.0, fs=30, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    filtered = bandpass(red_values.values, fs=fs)

    # Save result
    output_df = pd.DataFrame({
        'timestamp': timestamps,
        'ppg_signal': filtered
    })

    out_path = os.path.join('uploads', 'ppg_output.csv')
    output_df.to_csv(out_path, index=False)
    return out_path
