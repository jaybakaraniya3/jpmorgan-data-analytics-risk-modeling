import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("Nat_Gas (1).csv")
df.columns = ["Date", "Price"]

# Fix date format
df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%y')
df = df.sort_values("Date")

# ----------------------------
# 2. FEATURE ENGINEERING
# ----------------------------
df["Days"] = (df["Date"] - df["Date"].min()).dt.days
df["Sin"] = np.sin(2 * np.pi * df["Days"] / 365)
df["Cos"] = np.cos(2 * np.pi * df["Days"] / 365)

# ----------------------------
# 3. TRAIN MODEL
# ----------------------------
X = df[["Days", "Sin", "Cos"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# 4. PRICE PREDICTION FUNCTION
# ----------------------------
def predict_price(input_date):
    input_date = pd.to_datetime(input_date)
    days = (input_date - df["Date"].min()).days
    sin = np.sin(2 * np.pi * days / 365)
    cos = np.cos(2 * np.pi * days / 365)
    return float(model.predict([[days, sin, cos]])[0])

# ----------------------------
# 5. ADVANCED CONTRACT PRICING
# ----------------------------
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_storage,
    storage_cost_per_month
):
    storage = 0
    total_cost = 0
    total_revenue = 0

    injection_dates = sorted(pd.to_datetime(injection_dates))
    withdrawal_dates = sorted(pd.to_datetime(withdrawal_dates))

    # Injection (buy)
    for date in injection_dates:
        price = predict_price(date)
        inject_volume = min(injection_rate, max_storage - storage)
        storage += inject_volume
        total_cost += inject_volume * price

    # Withdrawal (sell)
    for date in withdrawal_dates:
        price = predict_price(date)
        withdraw_volume = min(withdrawal_rate, storage)
        storage -= withdraw_volume
        total_revenue += withdraw_volume * price

    # Storage cost
    if injection_dates and withdrawal_dates:
        start = min(injection_dates)
        end = max(withdrawal_dates)
        months = (end.year - start.year) * 12 + (end.month - start.month)
        storage_cost = storage_cost_per_month * months
    else:
        storage_cost = 0

    contract_value = total_revenue - total_cost - storage_cost

    return {
        "Total Injection Cost": total_cost,
        "Total Revenue": total_revenue,
        "Storage Cost": storage_cost,
        "Final Contract Value": contract_value
    }

# ----------------------------
# 6. TEST CASE
# ----------------------------
if __name__ == "__main__":
    result = price_storage_contract(
        injection_dates=["2024-05-31", "2024-06-30", "2024-07-31"],
        withdrawal_dates=["2024-12-31", "2025-01-31"],
        injection_rate=300000,
        withdrawal_rate=400000,
        max_storage=1000000,
        storage_cost_per_month=100000
    )

    print("\n--- CONTRACT VALUATION ---")
    for key, value in result.items():
        print(f"{key}: {value:.2f}")