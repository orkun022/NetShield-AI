"""
IDS Model Testleri — NetShield-AI
==================================
Preprocessing ve model pipeline testleri.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_generate_dataset():
    """Demo veri seti oluşturulabiliyor mu?"""
    from src.generate_dataset import generate_and_save
    filepath = generate_and_save(n_normal=20, n_dos=10, n_probe=10, n_r2l=5, n_u2r=5)
    assert os.path.exists(filepath), "Veri seti dosyası oluşturulmalı"

    df = pd.read_csv(filepath)
    assert len(df) == 50, f"50 satır olmalı, {len(df)} bulundu"
    assert 'label' in df.columns, "'label' sütunu olmalı"
    assert 'protocol_type' in df.columns, "'protocol_type' sütunu olmalı"


def test_preprocessing_runs():
    """Preprocessing pipeline çalışıyor mu?"""
    from src.preprocessing import load_data, preprocess

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    assert X_train.shape[0] > 0, "Eğitim verisi boş olmamalı"
    assert X_test.shape[0] > 0, "Test verisi boş olmamalı"
    assert len(feature_names) > 0, "Feature isimleri olmalı"
    assert X_train.shape[1] == X_test.shape[1], "Train ve test feature sayısı eşit olmalı"


def test_labels_are_binary():
    """Etiketler binary (0/1) mi?"""
    from src.preprocessing import load_data, preprocess

    df = load_data()
    _, _, y_train, y_test, _, _ = preprocess(df)

    unique_train = set(np.unique(y_train))
    unique_test = set(np.unique(y_test))

    assert unique_train.issubset({0, 1}), f"Train etiketler 0/1 olmalı, {unique_train} bulundu"
    assert unique_test.issubset({0, 1}), f"Test etiketler 0/1 olmalı, {unique_test} bulundu"


def test_random_forest_can_fit():
    """Random Forest eğitilebiliyor mu?"""
    from sklearn.ensemble import RandomForestClassifier
    from src.preprocessing import load_data, preprocess

    df = load_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(df)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(p in [0, 1] for p in predictions)


def test_one_hot_encoding_applied():
    """One-Hot Encoding uygulanmış mı?"""
    from src.preprocessing import load_data, preprocess

    df = load_data()
    _, _, _, _, _, feature_names = preprocess(df)

    # One-Hot Encoding sonrası protocol_type_tcp, service_http gibi sütunlar olmalı
    ohe_features = [f for f in feature_names if '_' in f and any(
        f.startswith(cat) for cat in ['protocol_type', 'service', 'flag']
    )]
    assert len(ohe_features) > 0, "One-Hot Encoding uygulanmamış"


def test_metrics_work():
    """Metrikler hesaplanabiliyor mu?"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from src.preprocessing import load_data, preprocess

    df = load_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(df)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1
