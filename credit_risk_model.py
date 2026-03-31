import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# ----------------------------
# 2. DATA CLEANING
# ----------------------------
df = df.dropna()

# Target column
target_col = "default"

# Features & target
X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------
# 3. TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. TRAIN MODEL (tuned for stability)
# ----------------------------
model = LogisticRegression(max_iter=1000, C=0.5)
model.fit(X_train, y_train)

# ----------------------------
# 5. MODEL EVALUATION
# ----------------------------
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 6. PREDICT PD FUNCTION
# ----------------------------
def predict_pd(input_data):
    input_df = pd.DataFrame([input_data])

    # Ensure all columns match training data
    for col in X.columns:
        if col not in input_df:
            input_df[col] = 0

    # Maintain correct column order
    input_df = input_df[X.columns]

    pd_value = model.predict_proba(input_df)[0][1]
    return pd_value

# ----------------------------
# 7. EXPECTED LOSS FUNCTION
# ----------------------------
def calculate_expected_loss(input_data, loan_amount):
    PD = predict_pd(input_data)
    LGD = 0.9  # 90% loss (10% recovery)

    expected_loss = PD * LGD * loan_amount
    return PD, expected_loss

# ----------------------------
# 8. TEST CASE (BALANCED BORROWER)
# ----------------------------
if __name__ == "__main__":
    
    sample_borrower = {
        "credit_lines_outstanding": 3,
        "loan_amt_outstanding": 20000,
        "total_debt_outstanding": 25000,
        "income": 60000,
        "fico_score": 680
    }

    loan_amount = 25000

    pd_value, el_value = calculate_expected_loss(sample_borrower, loan_amount)

    print("\n--- SAMPLE RESULT ---")
    print(f"Probability of Default (PD): {pd_value:.6f}")
    print(f"Expected Loss (EL = PD × LGD × EAD): {el_value:.2f}")