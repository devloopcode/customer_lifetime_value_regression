# 🚗 Auto Insurance Customer Lifetime Value (CLV) Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)

A machine learning regression project to predict **Customer Lifetime Value (CLV)** in the auto insurance domain.

---

## 📊 Dataset

- **Source:** [auto_insurance.csv](auto_insurance.csv)
- **Records:** 9,134 policyholder records
- **Target:** `Customer Lifetime Value` (continuous, USD)

---

## 📁 Project Structure

```
customer_lifetime_value_regression/
│
├── auto_insurance.csv                # Raw dataset
├── customer_lifetime_value.ipynb     # EDA, preprocessing & modeling pipeline
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- `pip` or `conda`

### Setup

```bash
git clone git@github.com:devloopcode/customer_lifetime_value_regression.git
cd customer_lifetime_value_regression

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run

```bash
jupyter notebook customer_lifetime_value.ipynb
```

---

## 👤 Author

- **Mohamed Idbenouakrim** - [@devloopcode](https://github.com/devloopcode)