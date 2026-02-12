"""
Model Eğitim Modülü — NetShield-AI (IDS)
==========================================
Random Forest sınıflandırıcısı ile ağ trafiği anomali tespiti.
Confusion Matrix, Precision, Recall, F1-Score değerlerini hesaplar.

────────────────────────────────────────────────────────────
DENETİMLİ (SUPERVISED) vs DENETİMSİZ (UNSUPERVISED) ÖĞRENME
────────────────────────────────────────────────────────────

Bu IDS projesi için iki temel yaklaşım değerlendirilebilir:

┌═══════════════════════════════════════════════════════════════════┐
│                  DENETİMLİ ÖĞRENME (Supervised)                  │
│                  ─────────────────────────────────                │
│  Algoritma: Random Forest, SVM, XGBoost                         │
│                                                                   │
│  ✅ Avantajlar:                                                  │
│    • Bilinen saldırı türlerini çok yüksek doğrulukla tespit eder│
│    • Sınıflandırma metrikleri (Precision, Recall) güvenilir     │
│    • Etiketli veri varsa en iyi performansı verir               │
│    • Saldırı türlerini kategorize edebilir (DoS, Probe, R2L)    │
│                                                                   │
│  ❌ Dezavantajlar:                                               │
│    • Etiketli (labeled) veri gerektirir — pahalı ve zaman alıcı │
│    • Yeni/bilinmeyen saldırı türlerini (zero-day) tanıyamaz     │
│    • Eğitim verisindeki dağılıma bağımlıdır                     │
├═══════════════════════════════════════════════════════════════════┤
│                 DENETİMSİZ ÖĞRENME (Unsupervised)                │
│                 ──────────────────────────────────                │
│  Algoritma: Isolation Forest, One-Class SVM, Autoencoder        │
│                                                                   │
│  ✅ Avantajlar:                                                  │
│    • Etiketli veriye ihtiyaç duymaz — sadece "normal" veri yeter│
│    • Zero-day saldırıları tespit edebilir (anomali = bilinmeyen)│
│    • Daha az veri toplama maliyeti                               │
│                                                                   │
│  ❌ Dezavantajlar:                                               │
│    • Daha yüksek False Positive oranı (normal trafiği saldırı   │
│      olarak işaretleyebilir)                                     │
│    • Saldırı türünü belirleyemez, sadece "anomali" der          │
│    • Eşik değeri (threshold) ayarı zordur                        │
└═══════════════════════════════════════════════════════════════════┘

BU PROJEDE:
  NSL-KDD veri seti ETİKETLİ olduğu için DENETİMLİ ÖĞRENME kullanıyoruz.
  Random Forest tercih edildi çünkü:
  1. Tabular veriler için en güvenilir algoritmalardan biridir
  2. Overfitting'e dirençlidir (bagging ensemble)
  3. Feature importance verdiği için hangi trafik özelliklerinin
     saldırı tespitinde önemli olduğunu gösterir
  4. Hem büyük hem küçük veri setlerinde iyi çalışır
────────────────────────────────────────────────────────────
"""

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
    """Confusion Matrix görselleştirmesi."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack'],
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')

    # ── False Negative Açıklaması ──
    # Confusion Matrix'teki 4 hücre:
    #   TN (True Negative)  : Normal trafiği doğru tespit
    #   FP (False Positive)  : Normal trafiği yanlışlıkla saldırı olarak işaretleme
    #   FN (False Negative) : SALDIRIYI KAÇIRMA — EN TEHLİKELİ DURUM!
    #   TP (True Positive)  : Saldırıyı doğru tespit
    fn = cm[1][0]  # Gerçek saldırı ama Normal diye tahmin edilen
    ax.text(0.5, -0.15,
            f'⚠️ False Negative (Kaçırılan Saldırı): {fn}',
            transform=ax.transAxes, fontsize=10, ha='center',
            color='red', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Kaydedildi: {path}")
    return cm


def plot_roc_curve(y_true, y_proba, model_name='Random Forest'):
    """ROC Curve görselleştirmesi."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Kaydedildi: {path}")
    return roc_auc


def plot_feature_importance(model, feature_names, model_name='Random Forest', top_n=20):
    """En önemli N feature'ı görselleştirir."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color=colors[::-1], edgecolor='white'
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance — {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Kaydedildi: {path}")


def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """
    Random Forest sınıflandırıcısını eğitir ve tüm metrikleri hesaplar.

    ────────────────────────────────────────────────────────────
    SİBER GÜVENLİKTE FALSE NEGATIVE'İN KRİTİK ÖNEMİ
    ────────────────────────────────────────────────────────────
    False Negative (FN) = Gerçek bir saldırıyı "Normal" olarak
    yanlış sınıflandırmak.

    Bu, siber güvenlikte EN TEHLİKELİ durumdur çünkü:

    1. SALDIRI TESPİT EDİLEMEZ: Saldırgan ağda fark edilmeden
       hareket eder, veri çalar veya sisteme zarar verir.

    2. GEÇ KALMA: Saldırı ancak hasar oluştuktan sonra fark edilir.
       IBM raporuna göre ortalama tespit süresi 277 gün!

    3. MADDİ KAYIP: Bir veri ihlalinin ortalama maliyeti $4.45M (2023).

    False Positive (FP) = Normal trafiği "Saldırı" olarak işaretlemek.
    Bu da kötüdür (alarm yorgunluğu) ama FN kadar tehlikeli DEĞİLDİR.

    SONUÇ: IDS sistemlerinde RECALL (True Positive Rate) metriği
    en önemli metriktir çünkü:
      Recall = TP / (TP + FN)
    Yüksek Recall = Düşük FN = Daha az kaçırılan saldırı.

    Bu yüzden modeli optimize ederken Accuracy'den çok RECALL'a
    odaklanmalıyız!
    ────────────────────────────────────────────────────────────
    """
    print("\n" + "=" * 60)
    print("🔧 RANDOM FOREST SINIFLANDIRICI EĞİTİMİ")
    print("=" * 60)

    # Model oluştur
    # n_estimators=200: 200 karar ağacı kullanılır (ensemble)
    # max_depth=20: Her ağacın maksimum derinliği (overfitting kontrolü)
    # n_jobs=-1: Tüm CPU çekirdeklerini kullan (paralel eğitim)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # ── Model Eğitimi ──
    # Eğitim verisi (%80) ile model eğitilir
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Eğitim süresi: {train_time:.2f}s")

    # ── Tahmin ──
    # Test verisi (%20) üzerinde tahmin yapılır
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Saldırı olasılığı

    # ── Performans Metrikleri ──
    # Sadece accuracy yetmez! Siber güvenlikte tüm metrikler önemlidir:
    acc = accuracy_score(y_test, y_pred)       # Doğruluk
    prec = precision_score(y_test, y_pred)     # Precision: TP / (TP + FP)
    rec = recall_score(y_test, y_pred)         # Recall: TP / (TP + FN) ← EN ÖNEMLİ!
    f1 = f1_score(y_test, y_pred)              # F1: Precision ve Recall'ın harmonik ortalaması

    print(f"\n  📊 Performans Metrikleri:")
    print(f"  {'─' * 35}")
    print(f"  Accuracy:    {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision:   {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  Recall:      {rec:.4f}  ({rec*100:.2f}%)")  # En önemli metrik!
    print(f"  F1-Score:    {f1:.4f}  ({f1*100:.2f}%)")

    # ── Cross-Validation ──
    # 5-Fold CV ile modelin genelleştirme yeteneğini test ediyoruz
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    print(f"  CV Recall:   {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # ── Classification Report ──
    # Her sınıf için ayrı ayrı metrikler
    print(f"\n  📋 Classification Report:")
    print("  " + "─" * 50)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
    for line in report.split('\n'):
        print(f"  {line}")

    # ── Confusion Matrix ──
    # Confusion Matrix'i hem yazdır hem görselleştir
    cm = plot_confusion_matrix(y_test, y_pred, 'Random Forest')
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  📐 Confusion Matrix Detayları:")
    print(f"  {'─' * 40}")
    print(f"  True Negative  (TN): {tn:>5} — Normal trafiği doğru tespit")
    print(f"  False Positive (FP): {fp:>5} — Normal ama saldırı dedi (yanlış alarm)")
    print(f"  False Negative (FN): {fn:>5} — SALDIRIYI KAÇIRDI! ⚠️")
    print(f"  True Positive  (TP): {tp:>5} — Saldırıyı doğru tespit ✅")

    # ⚠️ SİBER GÜVENLİKTE FALSE NEGATIVE KRİTİK UYARI ⚠️
    # False Negative (saldırıyı kaçırmak) en tehlikeli durumdur.
    # Çünkü tespit edilemeyen bir saldırgan ağda serbestçe hareket eder,
    # veri çalar veya sisteme kalıcı zarar verir.
    # IDS sistemlerinde FN oranının sıfıra yakın olması hedeflenir.
    # Bu yüzden Recall (=TP/(TP+FN)) en önemli metriktir.
    if fn > 0:
        print(f"\n  ⚠️ UYARI: {fn} saldırı kaçırıldı (False Negative)!")
        print(f"  Siber güvenlikte FN oranının düşük olması KRİTİKTİR.")
        print(f"  Kaçırılan her saldırı = Veri ihlali riski!")
    else:
        print(f"\n  ✅ Mükemmel! Hiçbir saldırı kaçırılmadı (FN=0).")

    # ── ROC Curve ──
    roc_auc = plot_roc_curve(y_test, y_proba, 'Random Forest')
    print(f"\n  ROC AUC: {roc_auc:.4f}")

    # ── Feature Importance ──
    plot_feature_importance(model, feature_names, 'Random Forest', top_n=min(20, len(feature_names)))

    # ── Model Kaydetme ──
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'random_forest.pkl')
    joblib.dump(model, model_path)
    print(f"\n  [✓] Model kaydedildi: {model_path}")

    return model, {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_recall_mean': cv_scores.mean(),
    }


def train_isolation_forest(X_train, X_test, y_test):
    """
    Isolation Forest ile denetimsiz anomali tespiti.
    Supervised ile karşılaştırma amacıyla eklendi.
    """
    print("\n" + "=" * 60)
    print("🔧 ISOLATION FOREST (DENETİMSİZ) KARŞILAŞTIRMA")
    print("=" * 60)

    # Isolation Forest — etiket gerektirmez, sadece anomalileri tespit eder
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.3,   # Verinin tahmini %30'u anomali
        random_state=42,
        n_jobs=-1,
    )

    # Sadece eğitim verisiyle fit et (etiket KULLANILMAZ)
    iso_model.fit(X_train)

    # Test verisinde tahmin: 1=normal, -1=anomali
    y_pred_iso = iso_model.predict(X_test)
    # -1=anomali → 1=saldırı, 1=normal → 0=normal
    y_pred_binary = np.where(y_pred_iso == -1, 1, 0)

    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary, zero_division=0)
    rec = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1-Score:    {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred_binary)
    plot_confusion_matrix(y_test, y_pred_binary, 'Isolation Forest')

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }


def main():
    """Ana eğitim akışı."""
    # 1. Veri yükle ve ön işle
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # 2. Random Forest (Denetimli)
    rf_model, rf_results = train_random_forest(X_train, X_test, y_train, y_test, feature_names)

    # 3. Isolation Forest (Denetimsiz — karşılaştırma)
    iso_results = train_isolation_forest(X_train, X_test, y_test)

    # 4. Karşılaştırma tablosu
    print("\n" + "=" * 60)
    print("📊 DENETİMLİ vs DENETİMSİZ KARŞILAŞTIRMA")
    print("=" * 60)
    print(f"\n{'Metrik':<15} {'Random Forest':>15} {'Isolation Forest':>18}")
    print("─" * 50)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        rf_val = rf_results[metric]
        iso_val = iso_results[metric]
        winner = '← ★' if rf_val > iso_val else ''
        print(f"{metric:<15} {rf_val:>14.4f} {iso_val:>17.4f} {winner}")

    print(f"\n✅ Sonuç: NSL-KDD etiketli veri seti için Denetimli Öğrenme")
    print(f"   (Random Forest) daha iyi performans gösterir.")
    print(f"   Özellikle RECALL metriğinde (saldırı kaçırma oranı)")
    print(f"   denetimli model üstündür.")

    print("\n" + "=" * 60)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 60)

    return rf_results, iso_results


if __name__ == '__main__':
    main()
