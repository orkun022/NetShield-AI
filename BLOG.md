# Building an AI-Based Network Intrusion Detection System

> A deep dive into NetShield-AI — why behavior beats signatures, and how Random Forest outperforms Isolation Forest for network security.

---

## The Problem: Why Traditional IDS Falls Short

Network intrusion detection has been dominated by **signature-based systems** like Snort and Suricata for decades. These tools work like antivirus software — they maintain a database of known attack patterns and flag matching traffic.

The fatal flaw? **They can't detect what they've never seen.**

With over **2,200 cyberattacks occurring daily** and the average cost of a data breach reaching **$4.45 million** (IBM, 2023), waiting for signature updates is no longer acceptable. NetShield-AI takes a fundamentally different approach: **behavior-based detection using machine learning**.

---

## The Dataset: NSL-KDD

I used the **NSL-KDD** benchmark dataset, the standard for IDS research since 2009. Each record represents a single network connection with **41 features**:

| Feature Category | Count | Examples |
|-----------------|-------|----------|
| **Basic TCP** | 9 | duration, protocol_type, service, flag, src_bytes |
| **Content** | 13 | num_failed_logins, logged_in, root_shell, num_access_files |
| **Time-based** | 9 | count, srv_count, same_srv_rate, diff_srv_rate |
| **Host-based** | 10 | dst_host_count, dst_host_srv_count, dst_host_same_srv_rate |

### Attack Categories

The dataset labels traffic as either **Normal** or one of **4 attack types**:

```
┌─────────────────────────────────────────────────────────────┐
│  DoS (Denial of Service)     │  Flood the target            │
│  Probe (Reconnaissance)      │  Scan for vulnerabilities    │
│  R2L (Remote to Local)       │  Unauthorized remote access  │
│  U2R (User to Root)          │  Privilege escalation        │
└─────────────────────────────────────────────────────────────┘
```

This makes it a **multi-class classification problem** — not just "normal vs attack", but identifying the specific attack type.

---

## The Experiment: Supervised vs Unsupervised

The core research question of NetShield-AI is: **Which learning paradigm is better suited for intrusion detection?**

### Approach 1: Supervised Learning (Random Forest)

Random Forest trains on **labeled data** — it learns the exact mapping from traffic features to attack labels.

```python
# Random Forest: 200 trees, trained on labeled NSL-KDD data
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # y_train contains attack labels
```

**Strengths:**
- Classifies attacks into specific categories (DoS, Probe, R2L, U2R)
- ~99% recall — catches virtually every attack
- Provides feature importance for interpretability

**Limitation:** Requires labeled training data. If a completely new attack type emerges, it may not generalize without retraining.

### Approach 2: Unsupervised Learning (Isolation Forest)

Isolation Forest learns what **normal traffic looks like**, then flags anything that deviates as an anomaly — no labels required.

```python
# Isolation Forest: no labels needed
if_model = IsolationForest(
    contamination=0.1,   # Assume ~10% of traffic is anomalous
    random_state=42,
    n_jobs=-1
)
if_model.fit(X_train_normal)  # Trained only on normal traffic
```

**Strengths:**
- No labeled data required — learns from normal traffic patterns
- Can theoretically detect **zero-day attacks** (completely new attack types)

**Limitation:** Can only say "normal" or "anomaly" — no attack categorization. Higher false positive rate.

---

## Results: Supervised Wins, But Unsupervised Has Its Place

| Metric | Random Forest | Isolation Forest |
|--------|---------------|------------------|
| **Accuracy** | ~99%+ | ~85-90% |
| **Precision** | ~99%+ | ~75-85% |
| **Recall** ⭐ | ~99%+ | ~80-90% |
| **F1-Score** | ~99%+ | ~78-87% |
| **Attack Classification** | ✅ 4 categories | ❌ Binary only |

### Why Recall Is the Most Critical Metric

In cybersecurity, a **False Negative** (missing a real attack) is far worse than a **False Positive** (flagging normal traffic as suspicious):

```
False Positive → Security team investigates a non-issue → Minor inconvenience
False Negative → Real attack goes undetected → Data breach → $4.45M average cost
```

Random Forest achieves ~99% recall, meaning it catches virtually every attack. Isolation Forest's ~85% recall means roughly **1 in 7 attacks could go undetected** — unacceptable for production security.

---

## Technical Decisions

### Why One-Hot Encoding over Label Encoding?

Three features in NSL-KDD are categorical: `protocol_type` (tcp/udp/icmp), `service` (http/ftp/smtp...), and `flag` (SF/S0/REJ...).

**Label Encoding** assigns numbers: tcp=0, udp=1, icmp=2. This creates a false ordinal relationship — the model might learn that `icmp > udp > tcp`, which is meaningless.

**One-Hot Encoding** creates binary columns: `protocol_tcp=1, protocol_udp=0, protocol_icmp=0`. This preserves the categorical nature without introducing false ordering.

### Why Random Forest over Gradient Boosting?

While XGBoost might squeeze slightly higher accuracy, Random Forest offers:
1. **Parallelism** — trees train independently (faster on multi-core)
2. **Robustness** — less prone to overfitting with noisy network data
3. **Interpretability** — feature importance directly maps to traffic patterns
4. **No hyperparameter sensitivity** — works well with default settings

### Why 5-Fold Cross-Validation?

Network traffic is highly variable. A single train/test split might not capture all attack patterns:
- Fold 1 might have more DoS attacks
- Fold 5 might have more Probe attacks

Cross-validation ensures the model performs consistently across all data distributions.

---

## Key Feature Insights

The top features identified by Random Forest reveal what matters most for intrusion detection:

| Rank | Feature | Why It Matters |
|------|---------|---------------|
| 1 | `src_bytes` | Attack traffic often has abnormal data volume |
| 2 | `dst_bytes` | Response size patterns differ for attacks |
| 3 | `same_srv_rate` | DoS attacks target same service repeatedly |
| 4 | `logged_in` | Many attacks occur without authentication |
| 5 | `count` | Connection frequency reveals scanning behavior |

---

## Deployment

The application runs as a **Streamlit web dashboard** where users can:
- Upload network traffic CSVs for analysis
- View real-time classification results
- Compare Random Forest vs Isolation Forest performance
- Explore feature importance rankings

```bash
streamlit run app/app.py
```

---

## Lessons Learned

1. **Supervised > Unsupervised** for labeled datasets — Random Forest's 99% recall vs Isolation Forest's 85% is decisive in cybersecurity
2. **Recall > Accuracy** — 99% accuracy means nothing if the 1% you miss is a real attack
3. **Feature engineering matters** — OHE vs Label Encoding made a measurable difference in model performance
4. **The real-world gap** — NSL-KDD is a benchmark; real production IDS would need continuous retraining with live traffic data

---

## What's Next

- Integration with real network interfaces (pcap capture)
- Deep Learning comparison (LSTM for sequential traffic patterns)
- Real-time streaming analysis with Apache Kafka
- CICIDS2017 dataset evaluation for modern attack types

---

*Built with Python, Scikit-learn, Random Forest, Isolation Forest, and Streamlit.*

**GitHub:** [github.com/orkun022/NetShield-AI](https://github.com/orkun022/NetShield-AI)
