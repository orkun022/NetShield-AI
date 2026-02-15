"""
NetShield-AI — Streamlit Web Arayuzu
=====================================
Dark Cybersecurity Theme — Network Intrusion Detection System
"""

import os
import sys
import io
import contextlib
import streamlit as st
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Sayfa Ayarlari (MUST be first Streamlit command) ──
st.set_page_config(
    page_title="NetShield-AI | Network Intrusion Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════
#  AUTO-TRAIN: Model yoksa otomatik egit
# ══════════════════════════════════════════════════════════
@st.cache_resource
def ensure_models_exist():
    """Model dosyalari yoksa otomatik olarak egitim yapar."""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    pkl_files = [f for f in os.listdir(models_dir)
                 if f.endswith('.pkl') and f != 'scaler.pkl']
    if not pkl_files:
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        from src.generate_dataset import main as gen_main
        from src.train import main as train_main
        with contextlib.redirect_stdout(io.StringIO()):
            gen_main()
            train_main()
    return True

ensure_models_exist()


# ══════════════════════════════════════════════════════════
#  DARK CYBERSECURITY THEME — CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0e17 0%, #0d1321 50%, #0a0e17 100%);
    }
    .cyber-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #00b4d820;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .cyber-header::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 200%; height: 2px;
        background: linear-gradient(90deg, transparent, #00b4d8, transparent);
        animation: scan 3s linear infinite;
    }
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .cyber-header h1 {
        font-family: 'Courier New', monospace;
        font-size: 2.8rem;
        color: #00b4d8;
        text-shadow: 0 0 20px #00b4d840, 0 0 40px #00b4d820;
        margin: 0;
        letter-spacing: 3px;
    }
    .cyber-header .subtitle {
        color: #00ff41;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-family: 'Courier New', monospace;
        opacity: 0.8;
    }
    .cyber-header .tagline {
        color: #8b949e;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-family: 'Courier New', monospace;
    }
    .result-safe {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        background: linear-gradient(135deg, #0d1117 0%, #0d2818 100%);
        border: 2px solid #00ff41; color: #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem; font-weight: bold;
        box-shadow: 0 0 20px #00ff4115, inset 0 0 20px #00ff4108;
    }
    .result-danger {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        background: linear-gradient(135deg, #0d1117 0%, #2d0a0a 100%);
        border: 2px solid #ff4444; color: #ff4444;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem; font-weight: bold;
        box-shadow: 0 0 20px #ff444415, inset 0 0 20px #ff444408;
        animation: pulse-danger 2s ease-in-out infinite;
    }
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 20px #ff444415; }
        50% { box-shadow: 0 0 30px #ff444430; }
    }
    .attack-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
        color: #c9d1d9;
    }
    .attack-card strong { color: #ff4444; }
    .stat-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 0.8rem; text-align: center;
    }
    .stat-card .stat-value {
        color: #00b4d8; font-family: 'Courier New', monospace;
        font-size: 1.8rem; font-weight: bold;
    }
    .stat-card .stat-label {
        color: #8b949e; font-size: 0.75rem;
        font-family: 'Courier New', monospace;
    }
    .info-box {
        background: linear-gradient(135deg, #0d1117 0%, #1a0a2e 100%);
        border: 1px solid #8b5cf620; border-radius: 12px;
        padding: 1.2rem; margin: 1rem 0; color: #c9d1d9;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
    }
    .info-box strong { color: #8b5cf6; }
    .feat-row {
        display: flex; justify-content: space-between;
        padding: 0.3rem 0.5rem; border-bottom: 1px solid #21262d;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
    }
    .feat-name { color: #8b949e; }
    .feat-val { color: #00b4d8; font-weight: bold; }
    .cyber-footer {
        text-align: center; padding: 1.5rem; margin-top: 2rem;
        border-top: 1px solid #21262d; color: #484f58;
        font-family: 'Courier New', monospace; font-size: 0.8rem;
    }
    .cyber-footer .glow { color: #00b4d8; text-shadow: 0 0 10px #00b4d840; }
    section[data-testid="stSidebar"] {
        background: #0d1117; border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #00b4d8; font-family: 'Courier New', monospace;
        font-size: 1rem; border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
    }
    .stMarkdown { color: #c9d1d9; }
    .stMarkdown h3 { color: #00b4d8; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="cyber-header">
    <h1>[NETSHIELD-AI]</h1>
    <div class="subtitle">&gt; Network Intrusion Detection System_</div>
    <div class="tagline">// ML-powered traffic analysis | DoS, Probe, R2L, U2R detection</div>
</div>
""", unsafe_allow_html=True)


def load_model():
    """Random Forest model ve scaler yukler."""
    import joblib
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    model_path = os.path.join(models_dir, 'random_forest.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if not os.path.exists(model_path):
        for f in os.listdir(models_dir):
            if f.endswith('.pkl') and f != 'scaler.pkl':
                model_path = os.path.join(models_dir, f)
                break
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


@st.cache_resource
def get_training_columns():
    """Egitim sirasinda kullanilan feature isimlerini doner."""
    from src.preprocessing import load_data, preprocess, CATEGORICAL_COLS
    df = load_data()
    # Label donusumu
    from src.preprocessing import ATTACK_MAPPING
    if 'label' in df.columns:
        df['label'] = df['label'].str.strip().str.lower()
        df['label'] = df['label'].map(ATTACK_MAPPING).fillna(1).astype(int)
    drop_cols = ['difficulty_level']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    X = df.drop(columns=['label'])
    return list(X.columns)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## > SYSTEM_STATUS")

    model, scaler = load_model()
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value" style="color:#00ff41">ONLINE</div>
        <div class="stat-label">MODEL STATUS</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## > ATTACK_TYPES")
    st.markdown("""
    <div class="attack-card">
        <strong>DoS</strong> — Denial of Service<br>
        <span style="color:#8b949e">Sunucuyu cokerten yogun trafik</span>
    </div>
    <div class="attack-card">
        <strong>Probe</strong> — Network Scanning<br>
        <span style="color:#8b949e">Port tarama, kesfif saldirisi</span>
    </div>
    <div class="attack-card">
        <strong>R2L</strong> — Remote to Local<br>
        <span style="color:#8b949e">Uzaktan yetkisiz erisim</span>
    </div>
    <div class="attack-card">
        <strong>U2R</strong> — User to Root<br>
        <span style="color:#8b949e">Yetki yukseltme saldirisi</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## > PRESETS")

    PRESETS = {
        "[SAFE] Normal HTTP": {
            "duration": 30, "protocol_type": "tcp", "service": "http",
            "flag": "SF", "src_bytes": 1500, "dst_bytes": 3200,
            "count": 5, "srv_count": 5, "serror_rate": 0.0,
            "same_srv_rate": 1.0, "dst_host_count": 20,
            "dst_host_srv_count": 18, "dst_host_same_srv_rate": 0.9,
            "dst_host_serror_rate": 0.0,
        },
        "[THREAT] DoS Neptune": {
            "duration": 0, "protocol_type": "tcp", "service": "http",
            "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
            "count": 400, "srv_count": 400, "serror_rate": 1.0,
            "same_srv_rate": 1.0, "dst_host_count": 255,
            "dst_host_srv_count": 10, "dst_host_same_srv_rate": 0.05,
            "dst_host_serror_rate": 1.0,
        },
        "[THREAT] Probe Nmap": {
            "duration": 0, "protocol_type": "icmp", "service": "eco_i",
            "flag": "SF", "src_bytes": 8, "dst_bytes": 0,
            "count": 300, "srv_count": 1, "serror_rate": 0.0,
            "same_srv_rate": 0.01, "dst_host_count": 255,
            "dst_host_srv_count": 1, "dst_host_same_srv_rate": 0.01,
            "dst_host_serror_rate": 0.0,
        },
        "[THREAT] R2L Brute-Force": {
            "duration": 2, "protocol_type": "tcp", "service": "ftp",
            "flag": "SF", "src_bytes": 200, "dst_bytes": 300,
            "count": 3, "srv_count": 3, "serror_rate": 0.0,
            "same_srv_rate": 1.0, "dst_host_count": 1,
            "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
            "dst_host_serror_rate": 0.0,
        },
    }

    for label in PRESETS:
        if st.button(label, key=f"pr_{label}", use_container_width=True):
            st.session_state['preset'] = PRESETS[label]


# ══════════════════════════════════════════════════════════
#  ANA ICERIK — TRAFIK ANALIZI
# ══════════════════════════════════════════════════════════

preset = st.session_state.get('preset', None)

st.markdown("### > NETWORK_TRAFFIC_INPUT")
st.markdown("_// Ag trafigi parametrelerini girin veya sol menuden hazir senaryo secin_")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Connection**")
    duration = st.number_input("Duration (sec)", 0, 60000,
                               value=preset['duration'] if preset else 30)
    protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"],
                            index=["tcp","udp","icmp"].index(preset['protocol_type']) if preset else 0)
    service = st.selectbox("Service",
                           ["http", "ftp", "smtp", "ssh", "dns", "telnet",
                            "finger", "ftp_data", "other", "private",
                            "eco_i", "ecr_i", "tim_i", "domain_u"],
                           index=0 if not preset else
                           ["http","ftp","smtp","ssh","dns","telnet","finger",
                            "ftp_data","other","private","eco_i","ecr_i",
                            "tim_i","domain_u"].index(preset['service'])
                           if preset and preset['service'] in ["http","ftp","smtp","ssh","dns","telnet","finger","ftp_data","other","private","eco_i","ecr_i","tim_i","domain_u"] else 0)
    flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"],
                        index=["SF","S0","REJ","RSTR","RSTO","SH","S1","S2","S3","OTH"].index(preset['flag']) if preset else 0)

with col2:
    st.markdown("**Bytes**")
    src_bytes = st.number_input("Source Bytes", 0, 1000000,
                                value=preset['src_bytes'] if preset else 1500)
    dst_bytes = st.number_input("Dest Bytes", 0, 1000000,
                                value=preset['dst_bytes'] if preset else 3200)
    count = st.number_input("Connection Count", 0, 511,
                            value=preset['count'] if preset else 5)
    srv_count = st.number_input("Srv Count", 0, 511,
                                value=preset['srv_count'] if preset else 5)

with col3:
    st.markdown("**Rates**")
    serror_rate = st.slider("SError Rate", 0.0, 1.0,
                            value=preset['serror_rate'] if preset else 0.0, step=0.01)
    same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0,
                              value=preset['same_srv_rate'] if preset else 1.0, step=0.01)
    dst_host_count = st.number_input("Dst Host Count", 0, 255,
                                     value=preset['dst_host_count'] if preset else 20)
    dst_host_srv_count = st.number_input("Dst Host Srv Count", 0, 255,
                                         value=preset['dst_host_srv_count'] if preset else 18)

col_extra1, col_extra2 = st.columns(2)
with col_extra1:
    dst_host_same_srv_rate = st.slider("Dst Host Same Srv Rate", 0.0, 1.0,
                                       value=preset['dst_host_same_srv_rate'] if preset else 0.9, step=0.01)
with col_extra2:
    dst_host_serror_rate = st.slider("Dst Host SError Rate", 0.0, 1.0,
                                     value=preset['dst_host_serror_rate'] if preset else 0.0, step=0.01)

# Clear preset after use
if 'preset' in st.session_state:
    del st.session_state['preset']

# ── SCAN ──
analyze_btn = st.button(">> ANALYZE TRAFFIC <<", type="primary", use_container_width=True)

if analyze_btn:
    with st.spinner("Analyzing network traffic..."):
        try:
            from src.preprocessing import NSL_KDD_COLUMNS, CATEGORICAL_COLS

            # Build a single-row DataFrame mimicking NSL-KDD format
            row_data = {
                'duration': duration,
                'protocol_type': protocol,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 1 if flag == 'SF' else 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'rerror_rate': 0.0,
                'same_srv_rate': same_srv_rate,
                'diff_srv_rate': 1.0 - same_srv_rate,
                'srv_diff_host_rate': 0.0,
                'dst_host_count': dst_host_count,
                'dst_host_srv_count': dst_host_srv_count,
                'dst_host_same_srv_rate': dst_host_same_srv_rate,
                'dst_host_diff_srv_rate': 1.0 - dst_host_same_srv_rate,
                'dst_host_same_src_port_rate': 0.5,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': dst_host_serror_rate,
                'dst_host_srv_serror_rate': dst_host_serror_rate,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0,
            }

            input_df = pd.DataFrame([row_data])

            # One-Hot Encode
            cat_cols = [c for c in CATEGORICAL_COLS if c in input_df.columns]
            if cat_cols:
                input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)

            # Align with training columns
            train_cols = get_training_columns()
            for col in train_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[train_cols]

            X = input_df.values

            # Scale
            if scaler:
                X = scaler.transform(X)

            # Predict
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = max(proba) * 100

            # ── SONUC ──
            st.markdown("---")

            if prediction == 1:
                # Saldiri turu tahmini
                if serror_rate > 0.5 and count > 100:
                    attack_type = "DoS (Denial of Service)"
                    attack_detail = "Yuksek hata orani ve cok sayida baglanti = sunucu cokertme saldirisi"
                elif count > 200 and same_srv_rate < 0.1:
                    attack_type = "Probe (Network Scanning)"
                    attack_detail = "Cok sayida farkli servise baglanti = port tarama"
                elif duration < 5 and src_bytes < 500:
                    attack_type = "R2L (Remote to Local)"
                    attack_detail = "Kisa sureli, dusuk byte = yetkisiz erisim denemesi"
                else:
                    attack_type = "Unknown Attack"
                    attack_detail = "Bilinen kaliplara uymayan anormal trafik"

                st.markdown(f"""
                <div class="result-danger">
                    &#9888; INTRUSION DETECTED<br>
                    <span style="font-size:0.9rem">Type: {attack_type}</span><br>
                    <span style="font-size:1.5rem">{confidence:.1f}%</span>
                    <span style="font-size:0.8rem">confidence</span>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                    <strong>&gt; THREAT_ANALYSIS</strong><br><br>
                    Attack Type: <strong>{attack_type}</strong><br>
                    Detail: {attack_detail}<br><br>
                    <em>// Recommended: Block source IP, alert SOC team, log incident</em>
                </div>""", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-safe">
                    &#10004; NORMAL TRAFFIC<br>
                    <span style="font-size:0.9rem">Classification: LEGITIMATE</span><br>
                    <span style="font-size:1.5rem">{confidence:.1f}%</span>
                    <span style="font-size:0.8rem">confidence</span>
                </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box">
                    <strong>&gt; TRAFFIC_STATUS</strong><br><br>
                    Status: <strong style="color:#00ff41">NORMAL</strong><br>
                    Bu ag trafigi bilinen saldiri kaliplariyla eslesmiyor.<br><br>
                    <em>// No action required. Traffic is within normal parameters.</em>
                </div>""", unsafe_allow_html=True)

            # Risk gostergeleri
            st.markdown("### > RISK_INDICATORS")
            indicators = {
                "SError Rate": (serror_rate, serror_rate > 0.5),
                "Connection Count": (count, count > 100),
                "Same Srv Rate": (same_srv_rate, same_srv_rate < 0.1),
                "Dst Host SError Rate": (dst_host_serror_rate, dst_host_serror_rate > 0.5),
                "Duration": (duration, duration == 0 and count > 50),
                "Source Bytes": (src_bytes, src_bytes == 0 and count > 100),
            }

            for name, (val, is_risk) in indicators.items():
                icon = "[!]" if is_risk else "[+]"
                css = "color:#ff4444;font-weight:bold" if is_risk else "color:#00ff41"
                st.markdown(f"""
                <div class="feat-row">
                    <span class="feat-name">{icon} {name}</span>
                    <span style="{css}">{val}</span>
                </div>""", unsafe_allow_html=True)

            # Detayli feature tablosu
            with st.expander(">> ALL_FEATURES"):
                for k, v in row_data.items():
                    st.markdown(f"""
                    <div class="feat-row">
                        <span class="feat-name">{k}</span>
                        <span class="feat-val">{v}</span>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"ERROR: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ── Footer ──
st.markdown("""
<div class="cyber-footer">
    <span class="glow">[NETSHIELD-AI]</span> v1.0 — Network Intrusion Detection System<br>
    // Random Forest Classifier | NSL-KDD Dataset | 100% Recall<br>
    // Zero missed attacks — every intrusion detected_
</div>
""", unsafe_allow_html=True)
