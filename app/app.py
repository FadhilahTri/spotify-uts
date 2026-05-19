from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lr_model = joblib.load(os.path.join(BASE_DIR, 'models', 'model_linear_regression.pkl'))
ann_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'model_ann.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

def predict_popularity(features):
    features_scaled = scaler.transform([features])
    lr_pred = lr_model.predict(features_scaled)[0]
    ann_pred = ann_model.predict(features_scaled, verbose=0)[0][0]
    return {
        'linear_regression': round(float(lr_pred), 2),
        'ann': round(float(ann_pred), 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        float(data['artist_popularity']),
        float(data['artist_followers']),
        float(data['track_duration_min']),
        float(data['track_number']),
        float(data['album_total_tracks']),
        int(data['explicit']),
        int(data['release_year']),
        int(data['album_type'])
    ]
    result = predict_popularity(features)
    return jsonify(result)

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)