from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import noisereduce as nr
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, fs, lowcut=0.7, highcut=4.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "PPG Processing Server is running"})

@app.route('/process', methods=['POST'])
def process_ppg():
    try:
        # Get and validate input
        red_data = request.get_json()
        if not red_data or not isinstance(red_data, list):
            return jsonify({"error": "Invalid input format"}), 400

        # Extract data
        timestamps = np.array([float(d.get('t', 0)) for d in red_data])
        red_values = np.array([float(d.get('r', 0)) for d in red_data])

        # Calculate sampling rate
        duration = timestamps[-1] - timestamps[0]
        n_points = len(timestamps)
        fs = n_points / duration if duration > 0 else 30.0

        # Resample to uniform timebase
        uniform_t = np.linspace(timestamps[0], timestamps[-1], n_points)
        interp = interp1d(timestamps, red_values, kind='linear', fill_value="extrapolate")
        resampled = interp(uniform_t)

        # Normalize
        normalized = (resampled - np.mean(resampled)) / np.std(resampled)

        # Noise reduction
        try:
            denoised = nr.reduce_noise(
                y=normalized,
                sr=fs,
                stationary=True,
                prop_decrease=0.95,
                time_mask_smooth_ms=500
            )
        except Exception as e:
            print(f"Noise reduction failed: {str(e)}")
            denoised = normalized

        # Bandpass filter
        filtered = bandpass_filter(denoised, fs)

        # Prepare response
        result = [{"t": round(float(t), 4), "ppg": round(float(v), 4)} 
                 for t, v in zip(uniform_t, filtered)]
        
        return jsonify({
            "status": "success",
            "fs": round(float(fs), 2),
            "data_points": len(result),
            "ppg_data": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
