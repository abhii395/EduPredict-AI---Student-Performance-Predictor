"""
app.py - Flask REST API + Static File Server
Student Performance Predictor
"""

import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)
CORS(app)

# ── Check if models are trained ────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODELS_READY = os.path.exists(os.path.join(MODEL_DIR, 'metadata.json'))


# ──────────────────────────────────────────────────────────────────────────────
# Frontend routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ──────────────────────────────────────────────────────────────────────────────
# API: Health check
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'models_ready': MODELS_READY,
        'message':      'Student Performance Predictor API is running.'
    })


# ──────────────────────────────────────────────────────────────────────────────
# API: Get model metrics
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/models', methods=['GET'])
def get_models():
    if not MODELS_READY:
        return jsonify({'error': 'Models not trained yet. Run train.py first.'}), 503
    try:
        from predict import get_model_metrics
        return jsonify(get_model_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: Get feature options
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/features', methods=['GET'])
def get_features():
    if not MODELS_READY:
        return jsonify({'error': 'Models not trained yet. Run train.py first.'}), 503
    try:
        from predict import get_feature_info
        return jsonify(get_feature_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: Predict
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODELS_READY:
        return jsonify({'error': 'Models not trained yet. Run train.py first.'}), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided.'}), 400

        required_fields = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities', 'study_hours', 'attendance_rate', 'past_exam_score']
        missing = [f for f in required_fields if f not in data or data[f] == '']
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        from predict import run_prediction
        result = run_prediction(data)
        return jsonify({'success': True, 'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: AI Insights
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/insights', methods=['POST'])
def insights():
    try:
        data = request.get_json()
        student_data = data.get('student_data', {})
        prediction   = data.get('prediction', {})

        from ai_insights import generate_insights
        insight_text = generate_insights(student_data, prediction)
        return jsonify({'success': True, 'insights': insight_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: List available plots
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/plots', methods=['GET'])
def list_plots():
    plots_dir = os.path.join(BASE_DIR, 'static', 'plots')
    if not os.path.exists(plots_dir):
        return jsonify({'plots': []})
    files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    return jsonify({'plots': [f'/static/plots/{f}' for f in sorted(files)]})


# ──────────────────────────────────────────────────────────────────────────────
# API: Dataset stats
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/stats', methods=['GET'])
def dataset_stats():
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'StudentsPerformance.csv'))
        avg_math    = round(df['math score'].mean(), 1)
        avg_reading = round(df['reading score'].mean(), 1)
        avg_writing = round(df['writing score'].mean(), 1)
        total       = len(df)
        avg_overall = round((df['math score'] + df['reading score'] + df['writing score']).mean() / 3, 1)
        pass_rate   = round(((df['math score'] + df['reading score'] + df['writing score']) / 3 >= 60).mean() * 100, 1)
        return jsonify({
            'total_students': total,
            'avg_math':       avg_math,
            'avg_reading':    avg_reading,
            'avg_writing':    avg_writing,
            'avg_overall':    avg_overall,
            'pass_rate':      pass_rate,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Student Performance Predictor - Flask Server")
    print("=" * 55)
    if not MODELS_READY:
        print("  [WARN] Models not found. Run `python train.py` first!")
    else:
        print("  [OK] Models loaded and ready.")
    print("  [INFO] Running at http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000, host='0.0.0.0')
