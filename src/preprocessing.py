"""
Veri Ön İşleme Modülü — NetShield-AI (IDS)
=============================================
NSL-KDD / CICIDS2017 formatındaki ağ trafiği verisini yükler,
kategorik verileri dönüştürür ve ML modeline hazırlar.

────────────────────────────────────────────────────────────
ONE-HOT ENCODING vs LABEL ENCODING — KARŞILAŞTIRMA
────────────────────────────────────────────────────────────

IDS veri setlerinde 'protocol_type' (tcp, udp, icmp), 'service' (http, ftp)
ve 'flag' (SF, S0, REJ) gibi KATEGORİK sütunlar bulunur.

Bu verileri sayısala çevirmek için iki yaklaşım vardır:

┌──────────────────┬──────────────────────────────────────────┐
│ LABEL ENCODING   │ Her kategoriye bir sayı atar:             │
│                  │   tcp=0, udp=1, icmp=2                   │
│                  │                                          │
│ ✅ Avantaj:      │ Basit, sütun sayısını artırmaz           │
│ ❌ Dezavantaj:   │ Sayısal sıralama (ordering) ima eder!    │
│                  │ Model "udp(1) > tcp(0)" sanabilir ama    │
│                  │ protokoller arasında böyle bir sıra YOK.  │
│                  │ Tree-based modeller için sorun değil,     │
│                  │ ancak SVM/LR gibi modellere zarar verir.  │
├──────────────────┼──────────────────────────────────────────┤
│ ONE-HOT ENCODING │ Her kategori için yeni bir sütun oluşturur│
│                  │   tcp=[1,0,0], udp=[0,1,0], icmp=[0,0,1] │
│                  │                                          │
│ ✅ Avantaj:      │ Sıralama problemi YOK. Tüm modeller      │
│                  │ için güvenli. Doğru temsil sağlar.        │
│ ❌ Dezavantaj:   │ Kategori sayısı çoksa (100+), sütun       │
│                  │ sayısı patlar (curse of dimensionality).  │
└──────────────────┴──────────────────────────────────────────┘

BU PROJEDE:
  - 'protocol_type' (3 kategori) → ONE-HOT ENCODING ✅
  - 'service' (70+ kategori) → ONE-HOT ENCODING ✅
    (Random Forest seyrek veriyle iyi çalışır)
  - 'flag' (11 kategori) → ONE-HOT ENCODING ✅

SONUÇ: Farklı model türleriyle çalışacağımız için One-Hot Encoding
tercih ediyoruz. Bu, modelin kategoriler arasında yanlış bir
sıralama ilişkisi öğrenmesini engeller.
────────────────────────────────────────────────────────────
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NSL-KDD sütun isimleri (veri setinde header yoksa kullanılır)
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level',
]

# Kategorik sütunlar — bunlar One-Hot Encoding ile dönüştürülecek
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Saldırı türleri → ikili sınıflandırma eşlemesi
# NSL-KDD'de 39 farklı saldırı türü vardır, bunları binary yapalım
ATTACK_MAPPING = {
    'normal': 0,  # Normal trafik
    # DoS saldırıları
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1,
    'teardrop': 1, 'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
    # Probe saldırıları
    'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1, 'mscan': 1, 'saint': 1,
    # R2L saldırıları
    'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1,
    'spy': 1, 'warezclient': 1, 'warezmaster': 1, 'snmpgetattack': 1,
    'named': 1, 'xlock': 1, 'xsnoop': 1, 'sendmail': 1,
    'httptunnel': 1, 'worm': 1, 'snmpguess': 1,
    # U2R saldırıları
    'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1, 'rootkit': 1,
    'xterm': 1, 'ps': 1, 'sqlattack': 1,
}


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    NSL-KDD formatındaki veri setini yükler.

    Eğer veri seti yoksa, demo veri seti oluşturur.

    Parameters
    ----------
    filepath : str, optional
        CSV dosya yolu.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'raw', 'network_traffic.csv')

    if not os.path.exists(filepath):
        print("[!] Veri seti bulunamadı, demo veri seti oluşturuluyor...")
        from src.generate_dataset import generate_and_save
        generate_and_save()

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    print(f"[✓] Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> tuple:
    """
    Veriyi ön işler:
    1. Label encoding (saldırı türü → 0/1 binary)
    2. One-Hot Encoding (kategorik sütunlar)
    3. Train/Test split
    4. StandardScaler

    Parameters
    ----------
    df : pd.DataFrame
        Ham veri.
    test_size : float
        Test oranı.
    random_state : int
        Rastgelelik seed'i.
    scale : bool
        Ölçeklendirme yapılsın mı.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("\n" + "=" * 50)
    print("VERİ ÖN İŞLEME")
    print("=" * 50)

    # ── 1. Label (Etiket) Dönüşümü ──
    # Saldırı türlerini binary'ye dönüştür: 0=Normal, 1=Saldırı
    if 'label' in df.columns:
        df['label'] = df['label'].str.strip().str.lower()
        # Bilinmeyen saldırı türlerini 1 (saldırı) olarak kabul et
        df['label'] = df['label'].map(ATTACK_MAPPING).fillna(1).astype(int)
        print(f"[✓] Label dönüştürüldü (Binary: 0=Normal, 1=Saldırı)")
        print(f"    Sınıf dağılımı:\n{df['label'].value_counts().to_string()}")

    # Gereksiz sütunları çıkar
    drop_cols = ['difficulty_level']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── 2. One-Hot Encoding ──
    # Kategorik sütunları (protocol_type, service, flag) One-Hot Encoding
    # ile dönüştürüyoruz. Bu, modelin kategoriler arasında
    # yanlış bir sıralama ilişkisi öğrenmesini engeller.
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_cols:
        print(f"\n[*] One-Hot Encoding uygulanıyor: {cat_cols}")
        for col in cat_cols:
            n_unique = df[col].nunique()
            print(f"    {col}: {n_unique} benzersiz değer")

        # pd.get_dummies ile One-Hot (sparse=False için tüm sütunlar oluşturulur)
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        print(f"[✓] One-Hot Encoding sonrası sütun sayısı: {df.shape[1]}")

    # ── 3. X ve y ayırma ──
    y = df['label'].values
    X = df.drop(columns=['label'])
    feature_names = list(X.columns)
    X = X.values

    print(f"\n[✓] Feature sayısı: {len(feature_names)}")

    # ── 4. Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[✓] Train/Test split ({1-test_size:.0%}/{test_size:.0%}):")
    print(f"    Train: {X_train.shape[0]} örnek, Test: {X_test.shape[0]} örnek")

    # ── 5. Ölçeklendirme ──
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[✓] Scaler kaydedildi: {scaler_path}")

    # ── 6. İşlenmiş veriyi kaydet ──
    processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_data.csv')
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    print("=" * 50)
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == '__main__':
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
    print(f"\nSonuç: X_train={X_train.shape}, X_test={X_test.shape}")
