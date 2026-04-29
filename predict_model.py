"""
Car Price Prediction — Linear Regression vs Random Forest
Dataset: CarDekho Used Car Dataset (car_data.csv)

Compares two models using an expanded feature set for better accuracy.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==============================================================
#  STEP 1 - Load the Dataset
# ==============================================================

print("=" * 70)
print("   CAR PRICE PREDICTION - Model Comparison")
print("=" * 70)


try:
    df = pd.read_csv('car_data.csv')
    print(f"\n[OK] Dataset loaded successfully!")

    print(f"   Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(f"\n   Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("[ERROR] Error: 'car_data.csv' not found.")

    print("   Place the CarDekho dataset in the project directory.")
    exit()


# ==============================================================
#  STEP 2 - Data Cleaning
# ==============================================================

print("\n" + "-" * 70)

print("  STEP 2: Data Cleaning")
print("-" * 70)


# 2a. Drop irrelevant columns
irrelevant = ['Unnamed: 0', 'car_name', 'brand', 'model']
dropped = [c for c in irrelevant if c in df.columns]
df = df.drop(columns=dropped, errors='ignore')
print(f"Dropped: {dropped}")


# 2b. Handle missing values
null_count = df.isnull().sum().sum()
if null_count > 0:
    print(f"Found {null_count} null values - dropping affected rows ...")
    df = df.dropna()
print(f"[OK] Clean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

print(f"   Remaining columns: {list(df.columns)}")


# ==============================================================
#  STEP 3 - Feature Selection  (EXPANDED)
# ==============================================================

print("\n" + "-" * 70)

print("  STEP 3: Feature Selection (Expanded)")
print("-" * 70)


target = 'selling_price'

# Numerical features — added mileage, engine, max_power, seats
numerical_features = [
    'vehicle_age',   # older cars → lower price
    'km_driven',     # higher mileage → lower price
    'mileage',       # fuel efficiency (km/l)
    'engine',        # engine displacement (cc)
    'max_power',     # peak horsepower (bhp)
    'seats',         # seating capacity
]

# Categorical features
categorical_features = [
    'fuel_type',          # Petrol / Diesel / CNG / LPG / Electric
    'transmission_type',  # Manual / Automatic
    'seller_type',        # Individual / Dealer / Trustmark Dealer
]

all_features = numerical_features + categorical_features

# Validate
missing = [c for c in all_features + [target] if c not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    exit()

X = df[all_features].copy()
y = df[target].copy()

print(f"[OK] Target:  {target}")
print(f"[OK] Numerical features  ({len(numerical_features)}): {numerical_features}")
print(f"[OK] Categorical features ({len(categorical_features)}): {categorical_features}")

print(f"   Total samples: {X.shape[0]:,}")

# Quick stats on numerical features
print(f"\n   Numerical Feature Statistics:")
print(X[numerical_features].describe().round(2).to_string())


# ==============================================================
#  STEP 4 - Encode Categorical Features (One-Hot)
# ==============================================================

print("\n" + "-" * 70)

print("  STEP 4: Encoding Categorical Features")
print("-" * 70)


for col in categorical_features:
    unique_vals = X[col].unique()
    print(f"   {col:25s} -> {len(unique_vals)} categories: {list(unique_vals)}")


X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print(f"\n[OK] After encoding: {X.shape[1]} feature columns")

print(f"   {list(X.columns)}")


# ==============================================================
#  STEP 5 - Train-Test Split
# ==============================================================

print("\n" + "-" * 70)

print("  STEP 5: Train-Test Split (80 / 20)")
print("-" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[OK] Training set: {X_train.shape[0]:,} samples")
print(f"[OK] Testing set:  {X_test.shape[0]:,} samples")



# ==============================================================
#  STEP 6 - Train Both Models
# ==============================================================

print("\n" + "-" * 70)

print("  STEP 6: Training Models")
print("-" * 70)


# ── Model A: Linear Regression ──
print("\nTraining Linear Regression ...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("   [OK] Linear Regression trained.")


# ── Model B: Random Forest ──
print("\nTraining Random Forest (100 trees) ...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1         # use all CPU cores
)
rf_model.fit(X_train, y_train)
print("   [OK] Random Forest trained.")



# ==============================================================
#  STEP 7 - Predictions
# ==============================================================

print("\n" + "-" * 70)

print("  STEP 7: Generating Predictions")
print("-" * 70)


lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("[OK] Predictions generated for both models.")



# ==============================================================
#  STEP 8 - Model Evaluation & Comparison
# ==============================================================

def evaluate(name, y_true, y_pred):
    """Return a dict of evaluation metrics."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'Model': name, 'R2 Score': r2, 'MAE ($)': mae, 'RMSE ($)': rmse}


lr_metrics = evaluate('Linear Regression', y_test, lr_pred)
rf_metrics = evaluate('Random Forest',     y_test, rf_pred)

print("\n" + "=" * 70)
print("             MODEL COMPARISON RESULTS")
print("=" * 70)


comparison = pd.DataFrame([lr_metrics, rf_metrics])
comparison['R² (%)'] = (comparison['R² Score'] * 100).round(2)

# Nice formatted table
print(f"\n{'Metric':<30s} {'Linear Regression':>20s} {'Random Forest':>20s}")
print("-" * 70)

print(f"{'R² Score':<30s} {lr_metrics['R² Score']:>19.4f} {rf_metrics['R² Score']:>19.4f}")
print(f"{'R² Percentage':<30s} {lr_metrics['R² Score']*100:>18.2f}% {rf_metrics['R² Score']*100:>18.2f}%")
print(f"{'Mean Absolute Error (MAE)':<30s} {'₹{:,.0f}'.format(lr_metrics['MAE (₹)']):>20s} {'₹{:,.0f}'.format(rf_metrics['MAE (₹)']):>20s}")
print(f"{'Root Mean Sq Error (RMSE)':<30s} {'₹{:,.0f}'.format(lr_metrics['RMSE (₹)']):>20s} {'₹{:,.0f}'.format(rf_metrics['RMSE (₹)']):>20s}")

# Determine winner
winner = 'Random Forest' if rf_metrics['R² Score'] > lr_metrics['R² Score'] else 'Linear Regression'
improvement = abs(rf_metrics['R² Score'] - lr_metrics['R² Score']) * 100
print(f"\nWinner: {winner}  (+{improvement:.2f}% improvement)")



# ── Feature Importance (Random Forest) ──
print("\n" + "-" * 70)

print("  🌲 Random Forest — Feature Importance (Top 10)")
print("-" * 70)


importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
    bar = "#" * int(imp * 50)

    print(f"   {i:2d}. {feat:30s}  {imp:.4f}  {bar}")


# ── Sample Predictions Side-by-Side ──
print("\n" + "-" * 70)

print("  Sample Predictions — First 10 Test Samples")
print("-" * 70)


results = pd.DataFrame({
    'Actual ($)':     y_test.values,
    'LR Pred ($)':    lr_pred.round(0),
    'RF Pred ($)':    rf_pred.round(0),
    'LR Error ($)':   np.abs(y_test.values - lr_pred).round(0),
    'RF Error ($)':   np.abs(y_test.values - rf_pred).round(0),

})
results.index = range(1, len(results) + 1)
print(results.head(10).to_string())


# ── Single Car Prediction Example ──
print("\n" + "-" * 70)

print("  🔮 Example: Single Car Price Prediction")
print("-" * 70)


sample = X_test.iloc[0]
lr_single = lr_model.predict(sample.values.reshape(1, -1))[0]
rf_single = rf_model.predict(sample.values.reshape(1, -1))[0]
actual    = y_test.iloc[0]

print(f"   Input Features:")
for col_name, val in sample.items():
    print(f"      {col_name:30s} = {val}")

print(f"\n   {'':30s} {'Linear Reg':>14s}  {'Random Forest':>14s}")
print(f"   {'Predicted Price':30s} {'${:,.0f}'.format(lr_single):>14s}  {'${:,.0f}'.format(rf_single):>14s}")
print(f"   {'Actual Price':30s} {'${:,.0f}'.format(actual):>14s}  {'${:,.0f}'.format(actual):>14s}")
print(f"   {'Error':30s} {'${:,.0f}'.format(abs(actual-lr_single)):>14s}  {'${:,.0f}'.format(abs(actual-rf_single)):>14s}")


print("\n" + "=" * 70)
print("   [OK] Done! Both models trained and compared successfully.")
print("=" * 70)

