from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import noisereduce as nr
import os
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, fs, lowcut=0.7, highcut=4.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "python_version": "3.10.0"})

@app.route('/process', methods=['POST'])
def process_ppg():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid data format"}), 400

        # Extract data with validation
        try:
            timestamps = np.array([float(d.get('t', 0)) for d in data])
            red_values = np.array([float(d.get('r', 0)) for d in data])
        except Exception as e:
            return jsonify({"error": f"Data parsing error: {str(e)}"}), 400

        # Processing pipeline
        duration = timestamps[-1] - timestamps[0]
        n_points = len(timestamps)
        fs = max(n_points / duration, 30.0) if duration > 0 else 30.0

        uniform_t = np.linspace(timestamps[0], timestamps[-1], n_points)
        interp = interp1d(timestamps, red_values, kind='linear', fill_value="extrapolate")
        resampled = interp(uniform_t)
        normalized = (resampled - np.mean(resampled)) / np.std(resampled)

        # Noise reduction with fallback
        try:
            denoised = nr.reduce_noise(
                y=normalized,
                sr=fs,
                stationary=True,
                prop_decrease=0.95
            )
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            denoised = normalized

        filtered = bandpass_filter(denoised, fs)

        # Prepare response
        result = {
            "status": "success",
            "fs": float(fs),
            "data_points": n_points,
            "ppg": [{"t": round(float(t), 4), "v": round(float(v), 4)} 
                   for t, v in zip(uniform_t, filtered)]
        }
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
