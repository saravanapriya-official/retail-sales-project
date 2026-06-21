# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/retail_sales.csv")

print("Dataset:")
print(df.head())

# -------------------------
# Data Preparation
# -------------------------

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Create Sales Column
df["Sales"] = df["Quantity"] * df["Price"]

# Extract Month
df["Month"] = df["Date"].dt.month

# -------------------------
# Basic Analysis
# -------------------------

print("\nTotal Sales:")
print(df["Sales"].sum())

print("\nAverage Sales:")
print(df["Sales"].mean())

print("\nSales by Product:")
print(df.groupby("Product")["Sales"].sum())

# -------------------------
# Machine Learning
# -------------------------

X = df[["Quantity", "Price", "Month"]]

y = df["Sales"]

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# -------------------------
# New Prediction
# -------------------------

new_data = pd.DataFrame({
    "Quantity": [4],
    "Price": [30000],
    "Month": [6]
})

predicted_sales = model.predict(new_data)

print("\nPredicted Sales:")
print(predicted_sales[0])

print("\nProject Completed Successfully")