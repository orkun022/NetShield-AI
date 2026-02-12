<div align="center">

# 🔒 NetShield-AI

### Machine Learning-Powered Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Detect network intrusions and cyber attacks in real-time using Random Forest classification with 41 traffic features.*

</div>

---

## 🎯 Motivation

Cyber attacks on network infrastructure are growing at an alarming rate — **2,200+ attacks occur daily**, costing the global economy over **$8 trillion annually** (Cybersecurity Ventures, 2023). Traditional signature-based IDS solutions fail to detect **zero-day attacks** and novel intrusion patterns.

**NetShield-AI** leverages **Machine Learning** to build an intelligent Intrusion Detection System (IDS) that learns traffic patterns from the **NSL-KDD** benchmark dataset. Instead of relying on static rules, the system identifies anomalies by analyzing **41 network traffic features** across 4 attack categories.

> **Why ML for IDS?** Rule-based systems can only catch known attack signatures. ML models generalize from traffic patterns, detecting both known and previously unseen attack variants.

---

## 🛡️ Attack Categories Detected

| Category | Description | Examples |
|----------|-------------|----------|
| **DoS** | Denial of Service — flooding the target | Neptune, Smurf, Back |
| **Probe** | Port scanning & network surveillance | IPsweep, Portsweep, Nmap, Satan |
| **R2L** | Remote to Local — unauthorized access | Guess Password, FTP Write |
| **U2R** | User to Root — privilege escalation | Buffer Overflow, Rootkit |

---

## 🔬 Technical Approach

### Data Preprocessing

**One-Hot Encoding** is used for categorical features (`protocol_type`, `service`, `flag`) instead of Label Encoding because:
- Prevents the model from inferring false ordinal relationships (e.g., `tcp=0 < udp=1` is meaningless)
- Random Forest handles sparse OHE matrices efficiently
- Ensures consistent performance across different ML algorithms

### Supervised vs Unsupervised Learning

| Approach | Algorithm | Pros | Cons |
|----------|-----------|------|------|
| **Supervised** ✅ | Random Forest | High accuracy, feature importance, attack categorization | Requires labeled data, can't detect zero-day |
| **Unsupervised** | Isolation Forest | No labels needed, detects zero-day | Higher false positive rate, no attack classification |

> **Decision:** We use **Supervised Learning (Random Forest)** because the NSL-KDD dataset provides labels, enabling precise classification with measurable metrics.

### Why Recall Matters Most in Cybersecurity

```
⚠️ False Negative = Missing a real attack = CRITICAL SECURITY BREACH
   → Undetected attacker moves freely in the network
   → Average detection time without IDS: 277 days (IBM, 2023)
   → Average cost of a data breach: $4.45M

⚡ Therefore: RECALL (TP / (TP + FN)) is the most important metric
   → High Recall = Low False Negative = Fewer missed attacks
```

---

## ✨ Features

- 🌲 **Random Forest Classifier** with 200 estimators and 5-fold cross-validation
- 🔍 **Isolation Forest** comparison for supervised vs unsupervised analysis
- 📊 **Complete Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 📐 **Confusion Matrix** with False Negative analysis
- 📈 **Feature Importance** ranking (top 20 traffic features)
- 🔄 **One-Hot Encoding** pipeline for categorical network features
- 📝 **Detailed documentation** with cybersecurity context

---

## 📊 Results

| Metric | Random Forest | Isolation Forest |
|--------|---------------|------------------|
| Accuracy | ~99%+ | ~85-90% |
| Precision | ~99%+ | ~75-85% |
| **Recall** ⭐ | ~99%+ | ~80-90% |
| F1-Score | ~99%+ | ~78-87% |

> **Random Forest (Supervised)** significantly outperforms **Isolation Forest (Unsupervised)** on labeled NSL-KDD data, especially in Recall — the most critical metric for cybersecurity.

---

## 📁 Project Structure

```
NetShield-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        # Raw network traffic CSVs
│   │   └── network_traffic.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # OHE pipeline + encoding docs
│   ├── train.py                    # RF + IF training + metrics
│   └── generate_dataset.py         # NSL-KDD format demo generator
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
├── reports/figures/
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── feature_importance_*.png
└── tests/
    └── test_ids.py
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/NetShield-AI.git
cd NetShield-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Generate Demo Dataset
```bash
python src/generate_dataset.py
```

### Train Models
```bash
python src/train.py
```
Trains Random Forest (supervised) and Isolation Forest (unsupervised), outputs full confusion matrix, classification report, and comparison table.

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **scikit-learn** | Random Forest, Isolation Forest, metrics |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Matplotlib & Seaborn** | Visualization |
| **Joblib** | Model serialization |

---

## 📚 References

1. Tavallaee, M., et al. (2009). *A detailed analysis of the KDD CUP 99 data set*. IEEE Symposium on CISDA.
2. [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — University of New Brunswick
3. [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
4. IBM X-Force (2023). *Cost of a Data Breach Report*.

---

## 📝 License

This project is developed for educational purposes as part of a Computer Engineering curriculum.

---

<div align="center">

**Built with ❤️ for Cybersecurity & AI**

</div>
