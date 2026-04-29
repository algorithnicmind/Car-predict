"""
Car Price Prediction — Train & Save the Best Model (Random Forest)

This script trains a Random Forest Regressor on the CarDekho dataset,
evaluates its performance, and saves the trained model + metadata
for the Flask web application to use.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==============================================================
#  STEP 1 — Load & Clean
# ==============================================================

print("=" * 60)
print("  TRAINING CAR PRICE PREDICTION MODEL")
print("=" * 60)

df = pd.read_csv('car_data.csv')
print(f"\n[OK] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Drop irrelevant columns
irrelevant = ['Unnamed: 0', 'car_name', 'brand', 'model']
df = df.drop(columns=[c for c in irrelevant if c in df.columns], errors='ignore')

# Drop nulls
df = df.dropna()
print(f"[OK] After cleaning: {df.shape[0]:,} rows")


# ==============================================================
#  STEP 2 — Feature Selection
# ==============================================================

target = 'selling_price'

numerical_features = [
    'vehicle_age',
    'km_driven',
    'mileage',
    'engine',
    'max_power',
    'seats',
]

categorical_features = [
    'fuel_type',
    'transmission_type',
    'seller_type',
]

all_features = numerical_features + categorical_features

X = df[all_features].copy()
y = df[target].copy()

print(f"[OK] Features: {len(all_features)} ({len(numerical_features)} numerical + {len(categorical_features)} categorical)")


# ==============================================================
#  STEP 3 — Extract metadata for the frontend dropdowns
# ==============================================================

metadata = {
    'numerical_features': numerical_features,
    'categorical_features': {},
    'feature_stats': {},
}

for col in categorical_features:
    metadata['categorical_features'][col] = sorted(X[col].unique().tolist())

for col in numerical_features:
    metadata['feature_stats'][col] = {
        'min': float(X[col].min()),
        'max': float(X[col].max()),
        'mean': float(X[col].mean().round(2)),
        'median': float(X[col].median()),
    }


# ==============================================================
#  STEP 4 — One-Hot Encode & Train
# ==============================================================

X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
print(f"[OK] After encoding: {X_encoded.shape[1]} feature columns")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"\n   Training set : {X_train.shape[0]:,} samples")
print(f"   Testing set  : {X_test.shape[0]:,} samples")

print("\nTraining Random Forest (100 trees) ...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("[OK] Model trained!")


# ==============================================================
#  STEP 5 — Evaluate
# ==============================================================

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n{'Metric':<30s} {'Value':>15s}")
print("-" * 45)
print(f"{'R2 Score':<30s} {r2:>14.4f}")
print(f"{'R2 Percentage':<30s} {r2*100:>13.2f}%")
print(f"{'Mean Absolute Error':<30s} Rs.{mae:>11,.0f}")
print(f"{'Root Mean Sq Error':<30s} Rs.{rmse:>11,.0f}")

metadata['model_metrics'] = {
    'r2_score': round(r2, 4),
    'r2_percentage': round(r2 * 100, 2),
    'mae': round(mae, 2),
    'rmse': round(rmse, 2),
}


# ==============================================================
#  STEP 6 — Save model + metadata
# ==============================================================

# Save the column order so the Flask app can reconstruct features correctly
metadata['model_columns'] = X_encoded.columns.tolist()

joblib.dump(model, 'model.pkl')
print("\n[OK] Model saved -> model.pkl")

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("[OK] Metadata saved -> model_metadata.json")

print("\n" + "=" * 60)
print("  DONE! Model is ready for the web app.")
print("=" * 60)
