import os
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def process_and_retrain():
    print("=" * 60)
    print("  AUTOMATED USER DATA PROCESSING & RETRAINING")
    print("=" * 60)

    user_data_path = 'user_data.csv'
    car_data_path = 'car_data.csv'
    
    if not os.path.exists(user_data_path):
        print("[ERROR] user_data.csv not found.")
        return

    # 1. Read new user data
    user_df = pd.read_csv(user_data_path)
    if user_df.empty:
        print("[INFO] No new user data to process.")
        return

    print(f"[OK] Found {len(user_df)} new data points.")

    # 2. Append to main dataset
    car_df = pd.read_csv(car_data_path)
    combined_df = pd.concat([car_df, user_df], ignore_index=True)
    combined_df.to_csv(car_data_path, index=False)
    print(f"[OK] Main dataset updated: {len(combined_df):,} rows.")

    # 3. Retrain Model (Simplified version of train_model.py logic)
    print("\n[STEP] Starting retraining...")
    
    # Cleaning (same as train_model.py)
    df = combined_df.copy()
    irrelevant = ['Unnamed: 0', 'car_name', 'brand', 'model']
    df = df.drop(columns=[c for c in irrelevant if c in df.columns], errors='ignore')
    df = df.dropna()
    
    target = 'selling_price'
    numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    categorical_features = ['fuel_type', 'transmission_type', 'seller_type']
    
    X = df[numerical_features + categorical_features]
    y = df[target]
    
    # Preprocessing
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)
    
    # Evaluate
    y_pred = model.predict(X_encoded)
    r2 = r2_score(y, y_pred)
    print(f"[OK] Retraining complete. New R2 Score: {r2*100:.2f}%")

    # 4. Save artifacts
    joblib.dump(model, 'model.pkl')
    
    # Update metadata columns in case categories changed
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    metadata['model_columns'] = X_encoded.columns.tolist()
    metadata['r2_percentage'] = round(r2 * 100, 2)
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("[OK] Model and metadata updated.")

    # 5. Reset user_data.csv (keep only headers)
    with open(user_data_path, 'r') as f:
        headers = f.readline()
    with open(user_data_path, 'w') as f:
        f.write(headers)
    print("[OK] user_data.csv cleared.")

    # 6. Create status file
    with open('training_done.txt', 'w') as f:
        f.write(f"Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"New dataset size: {len(combined_df)} rows\n")
        f.write(f"New R2 Score: {r2*100:.2f}%\n")
    print("[OK] training_done.txt created.")
    
    print("\n" + "=" * 60)
    print("  DONE! User data integrated into the core model.")
    print("=" * 60)

if __name__ == "__main__":
    process_and_retrain()
