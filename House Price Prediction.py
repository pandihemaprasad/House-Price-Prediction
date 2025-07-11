
# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv("house_data.csv")

# 3. Explore & Clean Data
print("First 5 rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop columns with too many missing values (or fill them)
df = df.dropna(axis=1)

# 4. Select Features and Target
X = df.select_dtypes(include=[np.number]).drop("SalePrice", axis=1)
y = df["SalePrice"]

# 5. Handle Missing Values
X = X.fillna(X.mean())

# 6. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature Scaling (for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)

# 9. Evaluate Models
def evaluate_model(model, X_test, y_test, scaled=False):
    predictions = model.predict(X_test if not scaled else scaler.transform(X_test))
    print(f"\nModel: {model.__class__.__name__}")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R² Score:", r2_score(y_test, predictions))
    print("-" * 30)

evaluate_model(lr_model, X_test, y_test, scaled=True)
evaluate_model(rf_model, X_test, y_test)

# 10. Plot Prediction vs Actual
plt.figure(figsize=(8,5))
plt.scatter(y_test, rf_model.predict(X_test), alpha=0.6, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: Actual vs Predicted House Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.tight_layout()
plt.show()
