from flask import Flask, request, jsonify, send_file
import os
import csv
from ppg_utils import extract_ppg

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    upload_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(upload_path)

    output_path = extract_ppg(upload_path)  # returns path to processed CSV
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
