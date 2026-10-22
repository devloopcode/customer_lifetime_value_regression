# 🚗 Auto Insurance Customer Lifetime Value (CLV) Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)

An end-to-end machine learning regression project to predict **Customer Lifetime Value (CLV)** in the auto insurance domain. Accurately forecasting policyholder lifetime value enables insurance providers to optimize acquisition budgets, prioritize retention strategies, customize coverage offerings, and balance underwriting risk.

---

## 📌 Table of Contents

- [Business Objective](#-business-objective)
- [Dataset Summary](#-dataset-summary)
- [Key Insights](#-key-insights)
- [Data Pipeline & Preprocessing](#-data-pipeline--preprocessing)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Strategic Applications](#-strategic-applications)
- [Author](#-author)

---

## 🎯 Business Objective

Predicting Customer Lifetime Value (CLV) empowers insurers to shift from short-term transaction-based metrics to long-term relationship management:

- **Targeted Retention:** Proactively identify and retain high-value policyholders at risk of churning.
- **Acquisition Cost Optimization:** Allocate marketing budgets across sales channels based on projected customer value rather than initial policy premiums.
- **Personalized Offerings:** Match policy coverage tiers and renewal packages to customer lifetime potential.
- **Risk & Margin Alignment:** Balance premium pricing against claim severity to maintain healthy underwriting margins.

---

## 📊 Dataset Summary

The dataset ([auto_insurance.csv](auto_insurance.csv)) contains customer profiles spanning demographics, policy details, claims history, and acquisition channels.

- **Total Records:** 9,134 policyholder records (8,971 unique profiles after deduplication)
- **Target Variable:** `Customer Lifetime Value` (continuous numeric value in USD)
- **Feature Categories:**
  - **Demographics:** State, Education, Employment Status, Gender, Income, Marital Status
  - **Policy Details:** Coverage Tier, Policy Type, Number of Policies, Monthly Premium, Tenure
  - **Claims Activity:** Months Since Last Claim, Number of Open Complaints, Total Claim Amount
  - **Sales & Vehicle:** Sales Channel, Renew Offer Type, Vehicle Class, Vehicle Size

---

## 🔍 Key Insights

- **Right-Skewed Target:** Customer Lifetime Value exhibits a heavy right tail, with a core cluster of standard policies and a subset of high-value accounts. Robust loss functions or target transformations help mitigate outlier sensitivity during modeling.
- **Primary Value Drivers:** `Monthly Premium Auto` and `Number of Policies` are the strongest indicators of lifetime value.
- **Policy Multiplier Effect:** Customers holding multiple policies generate substantially higher lifetime value compared to single-policy holders, highlighting a prime opportunity for bundling campaigns.
- **High-Value Segments:** Luxury vehicle classes and premium coverage tiers demonstrate higher median CLV, whereas acquisition channels and employment categories exhibit relatively consistent value across groups.
- **Claims & Premium Correlation:** Claim amounts correlate moderately with monthly premiums and vehicle tier, reflecting higher repair costs for premium assets.

---

## ⚙️ Data Pipeline & Preprocessing

The data preparation workflow prepares features for predictive modeling while preventing data leakage:

1. **Data Cleaning & Deduplication:** Removed customer IDs and purged duplicate customer records.
2. **Train / Test Split:** Partitioned the clean dataset into an 80% training set and 20% holdout test set with reproducible random state.
3. **Date Feature Engineering:** Extracted `Effective Day` (elapsed days from baseline) from policy effective dates and dropped redundant date fields.
4. **Unified Column Transformation:**
   - **Numerical Features:** Scaled using `StandardScaler` (e.g., income, premium, claims, tenure, effective day).
   - **Categorical Features:** Encoded using `OneHotEncoder` with unseen category handling.
   - Generates a final model-ready matrix of 65 encoded features.

---

## 📁 Project Structure

```
customer_lifetime_value_regression/
│
├── auto_insurance.csv               # Raw dataset
├── customer_lifetime_value.ipynb   # EDA, preprocessing & regression modeling pipeline
├── requirements.txt                 # Project environment dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` or `conda`

### 1. Clone the Repository

```bash
git clone git@github.com:devloopcode/customer_lifetime_value_regression.git
cd customer_lifetime_value_regression
```

### 2. Environment Setup

```bash
# Using Python venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook customer_lifetime_value.ipynb
# or
jupyter lab
```

---

## 💼 Strategic Applications

- **Tiered Account Management:** Segment customers by predicted CLV to prioritize support queues and dedicate claims adjusters to high-tier policyholders.
- **Cross-Selling & Bundling:** Incentivize single-policy customers to add secondary vehicles or supplementary policies.
- **Dynamic Renewal Offers:** Tailor renewal packages and marketing channels based on customer lifetime responsiveness.
- **Proactive Churn Mitigation:** Automatically flag high-value policyholders filing complaints for priority resolution.

---

## 👤 Author

- **Mohamed Idbenouakrim** - [@devloopcode](https://github.com/devloopcode)