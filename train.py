"""
train.py - Model Training Pipeline for Student Performance Predictor
Trains both Regression and Classification models on the new custom dataset.
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'StudentsPerformance.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
PLOTS_DIR  = os.path.join(BASE_DIR, 'static', 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = ['#6C63FF', '#FF6584', '#43D8C9', '#FFD166', '#EF476F', '#06D6A0']
sns.set_theme(style='darkgrid', palette=PALETTE)
plt.rcParams.update({
    'figure.facecolor': '#0F0F1A',
    'axes.facecolor':   '#1A1A2E',
    'text.color':       '#EAEAEA',
    'axes.labelcolor':  '#EAEAEA',
    'xtick.color':      '#AAAAAA',
    'ytick.color':      '#AAAAAA',
    'grid.color':       '#2A2A3E',
    'axes.edgecolor':   '#2A2A3E',
    'font.family':      'DejaVu Sans',
})


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess():
    """Load CSV, encode categoricals, create target columns, split data."""
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[INFO] Columns: {list(df.columns)}")

    # ── Encode categorical columns ─────────────────────────────────────────
    categorical_cols = [
        'gender', 'parental_education', 'internet_access', 'extracurricular_activities'
    ]
    numeric_cols = [
        'study_hours', 'attendance_rate', 'past_exam_score'
    ]
    
    encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── Target engineering ─────────────────────────────────────────────────
    # Classification target: pass (score >= 60) = 1, fail = 0
    df_encoded['pass_fail'] = (df_encoded['final_score'] >= 60).astype(int)

    # ── Feature columns ────────────────────────────────────────────────────
    feature_cols = categorical_cols + numeric_cols

    X = df_encoded[feature_cols]
    y_reg   = df_encoded['final_score']
    y_clf   = df_encoded['pass_fail']

    # ── Scale ALL features (numeric + encoded categorical) ─────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_reg, y_clf, scaler, encoders, df, df_encoded, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# 2. EDA PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def generate_eda_plots(df, df_encoded):
    """Generate exploratory data analysis plots."""
    print("[INFO] Generating EDA plots...")

    # ── Score distributions ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0F0F1A')
    
    ax.hist(df['final_score'], bins=20, color='#6C63FF', alpha=0.85, edgecolor='#0F0F1A')
    ax.axvline(df['final_score'].mean(), color='#FF6584', linestyle='--', linewidth=2,
               label=f"Mean: {df['final_score'].mean():.1f}")
    ax.set_title('Final Score Distribution', fontsize=14, fontweight='bold', color='#EAEAEA')
    ax.set_xlabel('Score', color='#AAAAAA')
    ax.set_ylabel('Count', color='#AAAAAA')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'score_distributions.png'), dpi=120,
                bbox_inches='tight', facecolor='#0F0F1A')
    plt.close()

    # ── Correlation heatmap ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0F0F1A')
    numeric_df = df_encoded.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, linecolor='#0F0F1A', ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold',
                 color='#EAEAEA', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=120,
                bbox_inches='tight', facecolor='#0F0F1A')
    plt.close()

    # ── Study Hours vs Score ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0F0F1A')
    sns.scatterplot(x='study_hours', y='final_score', hue='pass_fail', data=df_encoded, 
                    palette=['#EF476F', '#06D6A0'], alpha=0.7, ax=ax)
    ax.set_title('Study Hours vs Final Score', fontsize=14, fontweight='bold', color='#EAEAEA')
    ax.set_xlabel('Study Hours per Week', color='#AAAAAA')
    ax.set_ylabel('Final Score', color='#AAAAAA')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'study_vs_score.png'), dpi=120,
                bbox_inches='tight', facecolor='#0F0F1A')
    plt.close()

    # ── Attendance vs Score ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0F0F1A')
    sns.scatterplot(x='attendance_rate', y='final_score', hue='pass_fail', data=df_encoded, 
                    palette=['#EF476F', '#06D6A0'], alpha=0.7, ax=ax)
    ax.set_title('Attendance Rate vs Final Score', fontsize=14, fontweight='bold', color='#EAEAEA')
    ax.set_xlabel('Attendance Rate (%)', color='#AAAAAA')
    ax.set_ylabel('Final Score', color='#AAAAAA')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'attendance_vs_score.png'), dpi=120,
                bbox_inches='tight', facecolor='#0F0F1A')
    plt.close()

    print("[INFO] EDA plots saved to static/plots/")


# ──────────────────────────────────────────────────────────────────────────────
# 3. REGRESSION MODELS
# ──────────────────────────────────────────────────────────────────────────────
def train_regression_models(X_train, X_test, y_train, y_test):
    """Train all regression models and return metrics."""
    models = {
        'Linear Regression':           LinearRegression(),
        'SVM (SVR)':                   SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        'Random Forest':               RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':           GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'XGBoost':                     XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42,
                                                    verbosity=0, eval_metric='rmse'),
    }

    results = {}
    trained  = {}

    for name, model in models.items():
        print(f"  [TRAIN] {name} (regression)...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        mae   = mean_absolute_error(y_test, y_pred)
        r2    = r2_score(y_test, y_pred)
        cv    = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

        results[name] = {'RMSE': round(rmse,3), 'MAE': round(mae,3),
                         'R2': round(r2,3), 'CV_R2': round(cv,3)}
        trained[name] = model
        print(f"      RMSE={rmse:.3f} | MAE={mae:.3f} | R²={r2:.3f} | CV_R²={cv:.3f}")

    return trained, results


# ──────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION MODELS
# ──────────────────────────────────────────────────────────────────────────────
def train_classification_models(X_train, X_test, y_train, y_test):
    """Train all classification models and return metrics."""
    models = {
        'Logistic Regression':         LogisticRegression(max_iter=1000, random_state=42),
        'SVM (SVC)':                   SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
        'Random Forest':               RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':           GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
        'XGBoost':                     XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42,
                                                     verbosity=0, eval_metric='logloss'),
    }

    results = {}
    trained  = {}

    for name, model in models.items():
        print(f"  [TRAIN] {name} (classification)...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

        results[name] = {'Accuracy': round(acc,4), 'F1': round(f1,4), 'CV_Acc': round(cv,4)}
        trained[name] = model
        print(f"      Acc={acc:.4f} | F1={f1:.4f} | CV_Acc={cv:.4f}")

    return trained, results


# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL COMPARISON PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(reg_results, clf_results):
    """Bar charts comparing model performances."""
    print("[INFO] Generating model comparison plots...")

    # ── Regression comparison ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0F0F1A')
    names = list(reg_results.keys())
    short = ['LR', 'SVM', 'RF', 'GBR', 'XGB']

    r2_vals   = [reg_results[n]['R2']   for n in names]
    rmse_vals = [reg_results[n]['RMSE'] for n in names]

    bars0 = axes[0].bar(short, r2_vals, color=PALETTE[:len(names)], alpha=0.85, edgecolor='#0F0F1A')
    axes[0].set_title('R² Score (Regression)', fontsize=13, fontweight='bold', color='#EAEAEA')
    axes[0].set_ylabel('R²', color='#AAAAAA')
    for bar, v in zip(bars0, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10, color='#EAEAEA')

    bars1 = axes[1].bar(short, rmse_vals, color=PALETTE[:len(names)], alpha=0.85, edgecolor='#0F0F1A')
    axes[1].set_title('RMSE (Regression)', fontsize=13, fontweight='bold', color='#EAEAEA')
    axes[1].set_ylabel('RMSE', color='#AAAAAA')
    for bar, v in zip(bars1, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=10, color='#EAEAEA')

    fig.suptitle('Regression Model Comparison', fontsize=15, fontweight='bold', color='#EAEAEA')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'regression_comparison.png'), dpi=120,
                bbox_inches='tight', facecolor='#0F0F1A')
    plt.close()

    # ── Classification comparison ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0F0F1A')
    clf_names  = list(clf_results.keys())
    clf_short  = ['LG', 'SVM', 'RF', 'GBC', 'XGB']
    acc_vals   = [clf_results[n]['Accuracy'] for n in clf_names]
    f1_vals    = [clf_results[n]['F1']       for n in clf_names]

    bars2 = axes[0].bar(clf_short, acc_vals, color=PALETTE[:len(clf_names)], alpha=0.85, edgecolor='#0F0F1A')
    axes[0].set_title('Accuracy (Classification)', fontsize=13, fontweight='bold', color='#EAEAEA')
    axes[0].set_ylabel('Accuracy', color='#AAAAAA')
    axes[0].set_ylim(0, 1.1)
    for bar, v in zip(bars2, acc_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10, color='#EAEAEA')

    bars3 = axes[1].bar(clf_short, f1_vals, color=PALETTE[:len(clf_names)], alpha=0.85, edgecolor='#0F0F1A')
    axes[1].set_title('F1 Score (Classification)', fontsize=13, fontweight='bold', color='#EAEAEA')
    axes[1].set_ylabel('F1 Score', color='#AAAAAA')
    axes[1].set_ylim(0, 1.1)
    for bar, v in zip(bars3, f1_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10, color='#EAEAEA')

    fig.suptitle('Classification Model Comparison', fontsize=15, fontweight='bold', color='#EAEAEA')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'classification_comparison.png'), dpi=120,
                bbox_inches='tight', facecolor='#0F0F1A')
    plt.close()

    print("[INFO] Model comparison plots saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE PLOT
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(reg_models, clf_models, feature_cols):
    """Plot feature importance from tree-based models."""
    print("[INFO] Generating feature importance plots...")

    for task, models_dict in [('Regression', reg_models), ('Classification', clf_models)]:
        for name in ['Random Forest', 'XGBoost']:
            model = models_dict.get(name)
            if model is None:
                continue
            importance = model.feature_importances_
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor('#0F0F1A')
            sorted_idx = np.argsort(importance)
            bars = ax.barh(
                [feature_cols[i] for i in sorted_idx],
                importance[sorted_idx],
                color=PALETTE[:len(feature_cols)],
                alpha=0.85, edgecolor='#0F0F1A'
            )
            ax.set_title(f'{name} Feature Importance ({task})', fontsize=13,
                         fontweight='bold', color='#EAEAEA')
            ax.set_xlabel('Importance', color='#AAAAAA')
            for bar, v in zip(bars, importance[sorted_idx]):
                ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{v:.3f}', va='center', fontsize=10, color='#EAEAEA')
            plt.tight_layout()
            fname = f'feature_importance_{task.lower()}_{name.replace(" ","_").replace("(","").replace(")","")}.png'
            plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=120,
                        bbox_inches='tight', facecolor='#0F0F1A')
            plt.close()

    print("[INFO] Feature importance plots saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 7. SAVE MODELS & METADATA
# ──────────────────────────────────────────────────────────────────────────────
def save_models(reg_models, clf_models, reg_results, clf_results,
                scaler, encoders, feature_cols, numeric_cols, categorical_cols):
    """Persist all models and metadata via joblib."""

    best_reg_name = max(reg_results, key=lambda n: reg_results[n]['R2'])
    best_clf_name = max(clf_results, key=lambda n: clf_results[n]['Accuracy'])
    print(f"[INFO] Best Regression : {best_reg_name} (R²={reg_results[best_reg_name]['R2']})")
    print(f"[INFO] Best Classifier : {best_clf_name} (Acc={clf_results[best_clf_name]['Accuracy']})")

    for name, model in reg_models.items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, os.path.join(MODEL_DIR, f'reg_{safe}.pkl'))

    for name, model in clf_models.items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, os.path.join(MODEL_DIR, f'clf_{safe}.pkl'))

    joblib.dump(reg_models[best_reg_name], os.path.join(MODEL_DIR, 'best_regressor.pkl'))
    joblib.dump(clf_models[best_clf_name], os.path.join(MODEL_DIR, 'best_classifier.pkl'))

    joblib.dump(scaler,   os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.pkl'))

    metadata = {
        'feature_cols':     feature_cols,
        'numeric_cols':     numeric_cols,
        'categorical_cols': categorical_cols,
        'best_regressor':   best_reg_name,
        'best_classifier':  best_clf_name,
        'regression_results':     reg_results,
        'classification_results': clf_results,
    }
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] All models and metadata saved to {MODEL_DIR}/")
    return best_reg_name, best_clf_name


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Student Performance Predictor — Training Pipeline")
    print("=" * 60)

    # 1. Load & preprocess
    X_scaled, y_reg, y_clf, scaler, encoders, df, df_enc, feature_cols = load_and_preprocess()

    # 2. EDA plots
    generate_eda_plots(df, df_enc)

    # 3. Train/test split
    Xtr, Xte, yr_tr, yr_te = train_test_split(X_scaled, y_reg,  test_size=0.2, random_state=42)
    Xctr, Xcte, yc_tr, yc_te = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

    # 4. Train models
    print("\n[REGRESSION MODELS]")
    reg_models, reg_results = train_regression_models(Xtr, Xte, yr_tr, yr_te)

    print("\n[CLASSIFICATION MODELS]")
    clf_models, clf_results = train_classification_models(Xctr, Xcte, yc_tr, yc_te)

    # 5. Comparison plots
    plot_model_comparison(reg_results, clf_results)

    # 6. Feature importance
    plot_feature_importance(reg_models, clf_models, feature_cols)

    # 7. Save
    categorical_cols = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities']
    numeric_cols = ['study_hours', 'attendance_rate', 'past_exam_score']
    
    save_models(reg_models, clf_models, reg_results, clf_results,
                scaler, encoders, feature_cols, numeric_cols, categorical_cols)

    print("\n" + "=" * 60)
    print("  Training complete! Run app.py to start the server.")
    print("=" * 60)
