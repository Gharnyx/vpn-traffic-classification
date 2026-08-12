# vpn-traffic-classification
Research on VPN vs Non-VPN traffic classification using NFStream
# VPN vs Non-VPN Traffic Classification using NFStream

**A Preliminary Investigation of VPN vs Non-VPN Traffic Classification Using NFStream Flow Statistics and Rule-Based Pseudo-Labeling**

This repository contains the code, results, and research related to classifying VPN and non-VPN network traffic using statistical flow features extracted by **NFStream**.

---

## Project Overview

Traditional methods like Deep Packet Inspection (DPI) and port-based detection fail on modern encrypted VPN traffic. This project explores a **privacy-preserving** approach that uses only statistical flow features (no payload inspection, no ports) to classify VPN vs non-VPN traffic.

The work is positioned as an **exploratory study** that honestly highlights the practical limitations of rule-based pseudo-labeling on real residential network traffic.

---

## Key Highlights

- Real residential network capture: **4,068,895 packets**
- Extracted **71 bidirectional statistical features** using NFStream
- Rule-based pseudo-labeling (no ports, no DPI)
- Random Forest + baseline models (Logistic Regression, SVM, XGBoost)
- 5-Fold Stratified Cross-Validation
- Ground-truth validation performed using separate controlled captures
- Honest analysis of label leakage and limitations

---

## Results Summary

| Metric                        | Value                          |
|------------------------------|--------------------------------|
| CV Accuracy (Random Forest)  | 95.84% ± 0.59%                |
| Macro F1-Score               | 0.8636 ± 0.0165               |
| Ground-Truth Agreement       | 51.40%                        |
| VPN Recall (Ground Truth)    | 1.19%                         |

> **Note:** High cross-validation accuracy on pseudo-labels does **not** translate to reliable real-world performance. The ground-truth validation revealed significant limitations of purely statistical pseudo-labeling.

---


---

## Tools & Technologies Used

- **Python 3**
- **NFStream** – Flow feature extraction
- **Pandas / NumPy**
- **Scikit-learn** – Random Forest, cross-validation, metrics
- **imbalanced-learn** – SMOTE
- **Matplotlib / Seaborn** – Visualization
- **Wireshark** – Traffic capture

---

## How to Run

1. Install required packages:
```bash
pip install nfstream pandas scikit-learn imbalanced-learn matplotlib seaborn

