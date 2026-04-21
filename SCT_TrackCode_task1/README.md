# 🏠 House Price Prediction using Linear Regression

## 📌 Overview
This project implements a **Linear Regression model** to predict house prices based on key features such as:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms

It demonstrates a fundamental supervised learning approach for regression problems.

---

## 🎯 Objective
To build a predictive model that estimates house prices with high accuracy using structured numerical data.

---

## 🧠 Model Used
- Linear Regression (from Scikit-learn)

---

## 📊 Dataset Requirements
The dataset must contain the following columns:
- `square_footage`
- `bedrooms`
- `bathrooms`
- `price` (target variable)

---

## ⚙️ Workflow
1. Load dataset using Pandas  
2. Select relevant features  
3. Split into training and testing sets  
4. Train Linear Regression model  
5. Evaluate using MAE, MSE, and R² Score  

---

## 🚀 How to Run
```bash
pip install pandas scikit-learn
python house_price_regression.py
