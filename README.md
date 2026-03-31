# JPMorgan Data Analytics & Risk Modeling Simulation

This project is based on a real-world simulation inspired by JPMorgan Chase's Quantitative Research tasks. It demonstrates end-to-end data analytics, forecasting, and risk modeling techniques applied to financial datasets.

---

## 📊 Project Overview

This repository contains multiple data analytics and machine learning models focused on:

- Natural Gas Price Forecasting
- Storage Contract Valuation
- Credit Risk Modeling (Probability of Default)
- FICO Score Quantization & Risk Bucketing

The goal is to simulate real-world financial decision-making using Python and data-driven approaches.

---

## 🚀 Key Features

### 1. 📈 Natural Gas Price Forecasting
- Time series modeling using regression + seasonality
- Predicts gas prices for future dates
- Visualizes historical and forecasted trends

### 2. 💰 Storage Contract Pricing
- Calculates contract value based on:
  - Buy/Sell prices
  - Storage costs
  - Injection & withdrawal constraints
- Simulates real commodity trading strategies

### 3. 🧠 Credit Risk Modeling
- Logistic Regression model to predict **Probability of Default (PD)**
- Calculates **Expected Loss (EL)** using:
- Helps estimate financial risk in loan portfolios

### 4. 🏦 FICO Score Quantization
- Converts continuous credit scores into categorical risk buckets
- Uses quantile-based binning for generalization
- Builds a PD model using rating-based features

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure
-- gas_storage_model.py
-- credit_risk_model.py
-- fico_quantization_model.py
-- Nat_Gas (1).csv
-- Task 3 and 4_Loan_Data.csv
-- README.md


---

## 📊 Sample Results

- Credit Risk Model Accuracy: **98.6%**
- PD Predictions behave as expected:
  - Lower FICO → Higher risk
  - Higher FICO → Lower risk

---

## 🧠 Key Learnings

- Time series forecasting with seasonality
- Financial modeling for trading decisions
- Credit risk analysis (PD, LGD, EL)
- Feature engineering & quantization techniques
- Building production-style data pipelines

---

## 📌 Business Impact

This project demonstrates how data analytics can be used to:
- Forecast commodity prices
- Evaluate trading strategies
- Predict loan defaults
- Support risk management decisions

---

## 👤 Author

**Jay Bakaraniya** 
Jr. Data Analyst  

---

## ⭐ Final Note

This project reflects real-world applications of data analytics in finance and showcases practical skills in predictive modeling, risk assessment, and decision-making.
