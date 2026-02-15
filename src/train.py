import os
import sys
import time
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_data, preprocess

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')


def plot_confusion_matrix(y_true, y_pred, model_name='Random Forest'):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')

    fn = cm[1][0]
    ax.text(0.5, -0.15, f'False Negative (Kacirilan Saldiri): {fn}',
            transform=ax.transAxes, fontsize=10, ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return cm


def plot_roc_curve(y_true, y_proba, model_name='Random Forest'):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_feature_importance(model, feature_names, model_name='Random Forest', top_n=20):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), importances[indices][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 60)
    print("RANDOM FOREST EGITIMI")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Egitim suresi: {train_time:.2f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    print(f"  CV Recall: {cv_scores.mean():.4f}")

    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
    for line in report.split('\n'):
        print(f"  {line}")

    cm = plot_confusion_matrix(y_test, y_pred, 'Random Forest')
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    if fn > 0:
        print(f"  UYARI: {fn} saldiri kacirdi!")
    else:
        print(f"  Hicbir saldiri kacirilmadi (FN=0)")

    roc_auc = plot_roc_curve(y_test, y_proba, 'Random Forest')
    plot_feature_importance(model, feature_names, 'Random Forest', top_n=min(20, len(feature_names)))

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'random_forest.pkl')
    joblib.dump(model, model_path)
    print(f"  [+] Model kaydedildi: {model_path}")

    return model, {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': roc_auc, 'cv_recall_mean': cv_scores.mean(),
    }


def train_isolation_forest(X_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("ISOLATION FOREST (DENETIMSIZ)")
    print("=" * 60)

    iso_model = IsolationForest(
        n_estimators=200, contamination=0.3, random_state=42, n_jobs=-1,
    )
    iso_model.fit(X_train)

    y_pred_iso = iso_model.predict(X_test)
    y_pred_binary = np.where(y_pred_iso == -1, 1, 0)

    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary, zero_division=0)
    rec = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    plot_confusion_matrix(y_test, y_pred_binary, 'Isolation Forest')

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    rf_model, rf_results = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    iso_results = train_isolation_forest(X_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("DENETIMLI vs DENETIMSIZ KARSILASTIRMA")
    print("=" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        rf_val = rf_results[metric]
        iso_val = iso_results[metric]
        winner = '<-' if rf_val > iso_val else ''
        print(f"  {metric:<15} RF: {rf_val:.4f}  IF: {iso_val:.4f} {winner}")

    print("\nEgitim tamamlandi!")
    return rf_results, iso_results


if __name__ == '__main__':
    main()
