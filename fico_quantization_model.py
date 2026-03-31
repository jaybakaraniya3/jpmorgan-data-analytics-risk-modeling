import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
df = df.dropna()

# ----------------------------
# 2. CREATE BUCKETS (QUANTIZATION)
# ----------------------------
def create_buckets(df, n_buckets):
    # Create quantile-based buckets
    df["bucket"], bins = pd.qcut(
        df["fico_score"],
        q=n_buckets,
        labels=False,
        retbins=True,
        duplicates='drop'
    )
    return df, bins

# ----------------------------
# 3. ASSIGN RATINGS
# ----------------------------
def assign_ratings(df):
    # Lower rating = better score
    max_bucket = df["bucket"].max()
    df["rating"] = max_bucket - df["bucket"]
    return df

# ----------------------------
# 4. TRAIN PD MODEL
# ----------------------------
def train_model(df):
    X = pd.get_dummies(df["rating"])
    y = df["default"]

    model = LogisticRegression()
    model.fit(X, y)

    return model, X.columns

# ----------------------------
# 5. PREDICT PD FROM FICO
# ----------------------------
def predict_pd(fico_score, bins, model, columns):
    # Find bucket
    bucket = np.digitize(fico_score, bins) - 1

    # Convert to rating
    max_bucket = len(bins) - 2
    rating = max_bucket - bucket

    # Create input
    input_data = {col: 0 for col in columns}

    if rating in input_data:
        input_data[rating] = 1

    input_df = pd.DataFrame([input_data])

    pd_value = model.predict_proba(input_df)[0][1]
    return pd_value, rating

# ----------------------------
# 6. RUN PIPELINE
# ----------------------------
if __name__ == "__main__":

    n_buckets = 5  # you can change this

    df, bins = create_buckets(df, n_buckets)
    df = assign_ratings(df)

    model, columns = train_model(df)

    print("\n--- BUCKET BOUNDARIES ---")
    print(bins)

    print("\n--- SAMPLE PREDICTIONS ---")
    test_scores = [500, 620, 700, 780, 820]

    for score in test_scores:
        pd_value, rating = predict_pd(score, bins, model, columns)
        print(f"FICO: {score} → Rating: {rating} → PD: {pd_value:.4f}")