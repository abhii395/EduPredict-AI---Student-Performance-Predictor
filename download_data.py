"""
download_data.py — Generate a synthetic Students Performance dataset based on specific fields.
Run this script once before training: python download_data.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'StudentsPerformance.csv')

def generate_synthetic():
    """
    Generate a realistic synthetic dataset based on the requested fields:
    - gender
    - parental education (High School, Intermediate, Bachelors, Masters, PhD, None)
    - study hours per week (Numeric, 0-40)
    - attendance rate (Numeric, 50-100)
    - past exam score (Numeric, 0-100)
    - internet access at home (Yes/No)
    - extracurricular activities (Yes/No)
    - Target: final_score (Numeric, 0-100)
    """
    np.random.seed(42)
    n = 1000

    # Categorical features
    genders = np.random.choice(['Female', 'Male'], n, p=[0.5, 0.5])
    parents = np.random.choice(
        ['None', 'High School', 'Intermediate', 'Bachelors', 'Masters', 'PhD'],
        n, p=[0.1, 0.25, 0.25, 0.2, 0.15, 0.05]
    )
    internet = np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15])
    extracurricular = np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4])

    # Numerical features
    # Normal distributions clipped to realistic ranges
    study_hours = np.clip(np.random.normal(15, 8, n), 0, 40).astype(int)
    attendance = np.clip(np.random.normal(85, 10, n), 30, 100).astype(int)
    past_score = np.clip(np.random.normal(70, 15, n), 0, 100).astype(int)

    # Score generation with realistic correlations
    # We will build a realistic formula and then add random noise
    
    edu_boost = {
        'None': -5,
        'High School': 0,
        'Intermediate': 3,
        'Bachelors': 8,
        'Masters': 12,
        'PhD': 15
    }
    internet_boost = {'Yes': 5, 'No': -5}
    extra_boost = {'Yes': 3, 'No': 0}
    
    def get_boost(arr, d):
        return np.array([d[v] for v in arr])

    # Base equation for final score
    # Past score has strong correlation (0.4)
    # Study hours (0.6 points per hour)
    # Attendance (0.3 points per percent)
    
    base_score = 10  # Starting baseline
    
    raw_score = (
        base_score 
        + (past_score * 0.4) 
        + (study_hours * 0.6) 
        + (attendance * 0.3)
        + get_boost(parents, edu_boost)
        + get_boost(internet, internet_boost)
        + get_boost(extracurricular, extra_boost)
        + np.random.normal(0, 7, n)  # Noise
    )
    
    final_score = np.clip(raw_score, 0, 100).astype(int)

    df = pd.DataFrame({
        'gender': genders,
        'parental_education': parents,
        'study_hours': study_hours,
        'attendance_rate': attendance,
        'past_exam_score': past_score,
        'internet_access': internet,
        'extracurricular_activities': extracurricular,
        'final_score': final_score,
    })

    df.to_csv(DATA_FILE, index=False)
    print('[INFO] Synthetic dataset generated: {} rows -> {}'.format(len(df), DATA_FILE))


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    print('[INFO] Generating synthetic dataset based on custom fields...')
    generate_synthetic()
    
    import pandas as pd
    df = pd.read_csv(DATA_FILE)
    print(f'[INFO] Dataset loaded: {df.shape[0]} rows, columns: {list(df.columns)}')
