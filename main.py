from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import noisereduce as nr

app = Flask(__name__)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, fs, lowcut=0.7, highcut=4.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

@app.route('/process_ppg', methods=['POST'])
def process_ppg():
    try:
        data = request.get_json()
        timestamps = np.array([float(d['t']) for d in data])
        red_values = np.array([float(d['r']) for d in data])

        duration = timestamps[-1] - timestamps[0]
        n_points = len(timestamps)
        fs = n_points / duration if duration > 0 else 30.0

        uniform_t = np.linspace(timestamps[0], timestamps[-1], n_points)
        interp = interp1d(timestamps, red_values, kind='linear', fill_value="extrapolate")
        resampled = interp(uniform_t)

        normalized = (resampled - np.mean(resampled)) / np.std(resampled)

        try:
            denoised = nr.reduce_noise(
                y=normalized,
                sr=fs,
                stationary=True,
                prop_decrease=0.95,
                time_mask_smooth_ms=500
            )
        except:
            denoised = normalized

        filtered = bandpass_filter(denoised, fs)

        ppg_data = [{"t": round(float(t), 4), "ppg": round(float(v), 4)} for t, v in zip(uniform_t, filtered)]
        return jsonify(ppg_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "PPG Server Running!"

if __name__ == '__main__':
    app.run(debug=True)
