import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

ATTACK_MAPPING = {
    'normal': 0,
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1,
    'teardrop': 1, 'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
    'ipsweep': 1, 'nmap': 1, 'portsweep': 1, 'satan': 1, 'mscan': 1, 'saint': 1,
    'ftp_write': 1, 'guess_passwd': 1, 'imap': 1, 'multihop': 1, 'phf': 1,
    'spy': 1, 'warezclient': 1, 'warezmaster': 1, 'snmpgetattack': 1,
    'named': 1, 'xlock': 1, 'xsnoop': 1, 'sendmail': 1,
    'httptunnel': 1, 'worm': 1, 'snmpguess': 1,
    'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1, 'rootkit': 1,
    'xterm': 1, 'ps': 1, 'sqlattack': 1,
}


def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, 'data', 'raw', 'network_traffic.csv')

    if not os.path.exists(filepath):
        print("[!] Veri seti bulunamadi, demo veri olusturuluyor...")
        from src.generate_dataset import generate_and_save
        generate_and_save()

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    print(f"[+] Veri yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
    return df


def preprocess(df, test_size=0.2, random_state=42, scale=True):
    print("\n" + "=" * 50)
    print("VERI ON ISLEME")
    print("=" * 50)

    if 'label' in df.columns:
        df['label'] = df['label'].str.strip().str.lower()
        df['label'] = df['label'].map(ATTACK_MAPPING).fillna(1).astype(int)
        print(f"[+] Label donusturuldu (0=Normal, 1=Saldiri)")

    drop_cols = ['difficulty_level']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        print(f"[+] One-Hot Encoding sonrasi sutun sayisi: {df.shape[1]}")

    y = df['label'].values
    X = df.drop(columns=['label'])
    feature_names = list(X.columns)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[+] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == '__main__':
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
    print(f"\nSonuc: X_train={X_train.shape}, X_test={X_test.shape}")
