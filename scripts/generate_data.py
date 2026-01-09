import os
import numpy as np
import pandas as pd

# Create folders
os.makedirs("data/raw", exist_ok=True)

np.random.seed(42)

N = 1_500_000
days = 90
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)

df = pd.DataFrame({
    "event_date": np.random.choice(dates, N),
    "campaign_id": np.random.randint(1, 2000, N),
    "impressions": np.random.poisson(120, N),
    "clicks": np.random.poisson(12, N),
    "conversions": np.random.binomial(1, 0.05, N),
    "revenue": np.random.gamma(2, 20, N),
})

# Inject anomalies (ground truth)
df["label_injected_anomaly"] = 0
anom_idx = np.random.choice(df.index, int(0.01 * N), replace=False)
df.loc[anom_idx, "clicks"] *= 10
df.loc[anom_idx, "revenue"] *= 6
df.loc[anom_idx, "label_injected_anomaly"] = 1

df.to_parquet("data/raw/events.parquet", index=False)

print("Created data/raw/events.parquet")
print("Rows:", len(df))
