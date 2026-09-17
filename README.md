# EduPredict AI - Student Performance Predictor

## Overview
EduPredict AI is a machine learning web application designed to predict and analyze student academic performance. By evaluating critical metrics such as study hours, attendance rates, past exam scores, and other socio-educational factors, the system employs advanced predictive models (including XGBoost and Gradient Boosting) to forecast final scores. 

Additionally, it integrates with Mistral AI to provide context-aware, personalized study recommendations based on the predicted outcomes and the student's unique profile.

## Features
- **Accurate Performance Prediction:** Predicts exact final grades (0-100) and Pass/Fail status.
- **Multiple ML Models:** Compares 10 different regression and classification models in real-time.
- **AI-Powered Insights:** Uses Mistral AI to generate personalized, empathetic study recommendations.
- **Data Analytics Dashboard:** Visualizes feature importance, score distributions, and model performance.
- **Modern UI:** Built with a responsive, glassmorphism design using Vanilla HTML/CSS/JS.

## Tech Stack
- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: scikit-learn, XGBoost, Pandas, Numpy
- **Generative AI**: Mistral AI SDK (`codestral-2508` model)
- **Visualization**: Matplotlib, Seaborn, Chart.js (Frontend)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript

## Project Structure
```text
student-performance-predictor/
├── app.py                 # Main Flask server and API endpoints
├── train.py               # ML model training and evaluation pipeline
├── predict.py             # Inference utility and data preprocessing
├── download_data.py       # Script to generate the synthetic dataset
├── ai_insights.py         # Mistral AI integration logic
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment variables
├── data/                  # Dataset directory
├── models/                # Saved ML models, scalers, and encoders (.pkl)
├── static/                # CSS, JS, and generated plot images
└── templates/             # HTML templates (index.html)
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/abhii395/EduPredict-AI---Student-Performance-Predictor
cd student-performance-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
You must set up your API keys. Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and add your real Mistral API key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```
> **Security Warning:** Never commit your `.env` file to version control. It is ignored in `.gitignore`.

## Running the Project

### 1. Generate Dataset
Run the data script to generate the required synthetic dataset:
```bash
python download_data.py
```

### 2. Train Models
Train the regression and classification models (this will also generate the analytics plots):
```bash
python train.py
```

### 3. Start the Server
Launch the Flask backend:
```bash
python app.py
```
The application will be accessible at `http://localhost:5000`.

## Model / Dataset Information
- **Dataset:** The project uses a synthetic dataset generated via `download_data.py` designed to mimic realistic correlations between study hours, attendance, past scores, and academic success.
- **Models:** The system trains several estimators (Linear/Logistic Regression, SVR/SVC, Random Forest, Gradient Boosting, XGBoost). The best performing models are automatically saved as `best_regressor.pkl` and `best_classifier.pkl` and loaded for inference.

## Security
This project uses sensitive API keys. 
- Ensure that your `.env` file is never committed. 
- The `.gitignore` is specifically configured to ignore `.env`, `__pycache__`, and other sensitive or environment-specific files.

## License
MIT License.
