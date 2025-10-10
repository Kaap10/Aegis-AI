import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Pools for user_ids and ip_addresses
USER_POOL = [f'user_{i:03d}' for i in range(1, 101)]
IP_POOL = [f'192.168.1.{i}' for i in range(1, 26)] + [f'172.16.0.{i}' for i in range(1, 25)]
# For impossible travel, use two distant IPs
DISTANT_IPS = ['49.37.16.1', '104.244.42.1']  # India, USA

EVENT_TYPES = ['LOGIN', 'API_CALL', 'LOGOUT']
STATUS = ['SUCCESS', 'FAILURE']


def random_timestamp(start, end):
    """Generate a random timestamp between start and end datetimes."""
    delta = end - start
    int_delta = int(delta.total_seconds())
    random_second = random.randint(0, int_delta)
    return start + timedelta(seconds=random_second)


def generate_logs(num_rows=5000):
    logs = []
    now = datetime.now()
    start_time = now - timedelta(days=7)

    # 95% normal, 5% anomalies
    num_normal = int(num_rows * 0.95)
    num_anomaly = num_rows - num_normal

    # Normal logs
    for _ in range(num_normal):
        timestamp = random_timestamp(start_time, now)
        user_id = random.choice(USER_POOL)
        ip_address = random.choice(IP_POOL)
        event_type = random.choices(EVENT_TYPES, weights=[0.5, 0.4, 0.1])[0]
        status = 'SUCCESS' if event_type != 'LOGIN' else random.choices(['SUCCESS', 'FAILURE'], weights=[0.95, 0.05])[0]
        logs.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'ip_address': ip_address,
            'event_type': event_type,
            'status': status
        })

    # Anomalies
    # 1. Brute-force attempts
    for _ in range(num_anomaly // 3):
        user_id = random.choice(USER_POOL)
        ip_address = random.choice(IP_POOL)
        base_time = random_timestamp(start_time, now)
        for i in range(5):
            timestamp = base_time + timedelta(seconds=i * random.randint(1, 10))
            logs.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': user_id,
                'ip_address': ip_address,
                'event_type': 'LOGIN',
                'status': 'FAILURE'
            })

    # 2. Impossible travel
    for _ in range(num_anomaly // 3):
        user_id = random.choice(USER_POOL)
        base_time = random_timestamp(start_time, now)
        # India login
        logs.append({
            'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'ip_address': DISTANT_IPS[0],
            'event_type': 'LOGIN',
            'status': 'SUCCESS'
        })
        # USA login within 2 minutes
        logs.append({
            'timestamp': (base_time + timedelta(seconds=random.randint(30, 120))).strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'ip_address': DISTANT_IPS[1],
            'event_type': 'LOGIN',
            'status': 'SUCCESS'
        })

    # 3. Unusual API activity
    for _ in range(num_anomaly // 3):
        user_id = random.choice(USER_POOL)
        ip_address = random.choice(IP_POOL)
        base_time = random_timestamp(start_time, now)
        for i in range(20):
            timestamp = base_time + timedelta(seconds=i * random.randint(1, 5))
            logs.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': user_id,
                'ip_address': ip_address,
                'event_type': 'API_CALL',
                'status': 'SUCCESS'
            })

    # Shuffle logs and trim to num_rows
    random.shuffle(logs)
    logs = logs[:num_rows]
    df = pd.DataFrame(logs)
    os.makedirs(os.path.join(os.path.dirname(__file__), '../../data'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '../../data/synthetic_auth_logs.csv')
    df.to_csv(out_path, index=False)
    print(f"Successfully generated {len(df)} log entries and saved to data/synthetic_auth_logs.csv")

if __name__ == "__main__":
    generate_logs()
