import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from datetime import timedelta
import os

# Define IP geolocations (expand as needed for your synthetic data)
IP_COORDINATES = {
    '192.168.1.1': (28.6139, 77.2090),   # New Delhi, India
    '192.168.1.2': (28.6139, 77.2090),
    '192.168.1.3': (28.6139, 77.2090),
    '172.16.0.1': (19.0760, 72.8777),   # Mumbai, India
    '172.16.0.2': (19.0760, 72.8777),
    '203.0.113.15': (34.0522, -118.2437),  # LA, USA
    '198.51.100.8': (51.5074, -0.1278),    # London, UK
    '8.8.8.8': (37.386, -122.0838),    # Google DNS (Mountain View)
    '49.37.16.1': (28.7041, 77.1025),    # Delhi-ish
    '104.244.42.1': (40.7128, -74.0060),   # NY, USA-ish
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def main():
    # Load data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/synthetic_auth_logs.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Feature 1: Impossible Travel Speed
    df = df.sort_values(['user_id', 'timestamp'])
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    df['prev_ip'] = df.groupby('user_id')['ip_address'].shift(1)

    def compute_speed(row):
        prev = row['prev_ip']
        curr = row['ip_address']
        if pd.isna(prev) or prev not in IP_COORDINATES or curr not in IP_COORDINATES:
            return 0.0
        time_diff = (row['timestamp'] - row['prev_timestamp']).total_seconds() / 3600
        if time_diff == 0:
            return 0.0
        lat1, lon1 = IP_COORDINATES[prev]
        lat2, lon2 = IP_COORDINATES[curr]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        return dist / time_diff if time_diff > 0 else 0.0

    mask_login_success = (df['event_type'] == 'LOGIN') & (df['status'] == 'SUCCESS')
    df['impossible_travel_speed'] = 0.0
    df.loc[mask_login_success, 'impossible_travel_speed'] = df[mask_login_success].apply(compute_speed, axis=1)

    # Feature 2: Login frequency in 1 hour
    df['login_frequency_1hr'] = 0
    for idx, row in df.iterrows():
        cut_time = row['timestamp'] - timedelta(hours=1)
        window = df[(df['user_id'] == row['user_id']) & (df['timestamp'] >= cut_time) & (df['timestamp'] < row['timestamp'])]
        df.at[idx, 'login_frequency_1hr'] = window.shape[0]

    # Feature 3: Unique IP change count in 24 hr
    df['ip_change_count_24hr'] = 0
    for idx, row in df.iterrows():
        cut_time = row['timestamp'] - timedelta(hours=24)
        window = df[(df['user_id'] == row['user_id']) & (df['timestamp'] >= cut_time) & (df['timestamp'] < row['timestamp'])]
        df.at[idx, 'ip_change_count_24hr'] = window['ip_address'].nunique()

    print("Feature engineering complete.")

    features = ['impossible_travel_speed', 'login_frequency_1hr', 'ip_change_count_24hr']
    # Use only normal data for training
    normal_data = df[(df['status'] == 'SUCCESS') & (df['impossible_travel_speed'] <= 1000)]
    X = normal_data[features]

    # Train IsolationForest
    clf = IsolationForest(contamination='auto', random_state=42)

    # --- THIS IS THE UPDATE ---
    # Print a summary of the data the model is being trained on.
    print("\n--- Training Data Summary ---")
    print(X.describe())
    print("---------------------------\n")
    # --------------------------

    clf.fit(X)
    print("IsolationForest model trained successfully.")

    # Save model
    os.makedirs(os.path.join(os.path.dirname(__file__), '../../models'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '../../models/anomaly_detector.pkl')
    joblib.dump(clf, model_path)
    print(f"Model saved to models/anomaly_detector.pkl")

if __name__ == "__main__":
    main()