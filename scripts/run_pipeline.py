import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.tsa.seasonal import STL

# Create output folders
os.makedirs("data/gold", exist_ok=True)
os.makedirs("runs", exist_ok=True)

# Load data
df = pd.read_parquet("data/raw/events.parquet")
df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

# Aggregate to campaign-day
camp_daily = df.groupby(["campaign_id", "event_date"], as_index=False).agg(
    impressions=("impressions", "sum"),
    clicks=("clicks", "sum"),
    conversions=("conversions", "sum"),
    revenue=("revenue", "sum"),
    label_injected_anomaly=("label_injected_anomaly", "max"),
)

X = camp_daily[["impressions","clicks","conversions","revenue"]]
y = camp_daily["label_injected_anomaly"]

# Isolation Forest
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)
iso_scores = -iso.fit(X).score_samples(X)
camp_daily["iso_score"] = iso_scores

# One-Class SVM
oc = OneClassSVM(nu=0.01, gamma="scale")
oc_scores = -oc.fit(X).decision_function(X)
camp_daily["ocsvm_score"] = oc_scores

# Metrics
metrics = {
    "isolation_forest": {
        "roc_auc": float(roc_auc_score(y, iso_scores)),
        "pr_auc": float(average_precision_score(y, iso_scores))
    },
    "oneclass_svm": {
        "roc_auc": float(roc_auc_score(y, oc_scores)),
        "pr_auc": float(average_precision_score(y, oc_scores))
    }
}

# Save outputs
camp_daily.to_parquet("data/gold/campaign_daily_anomalies.parquet", index=False)

# Time-series anomalies
daily = df.groupby("event_date", as_index=False).agg(revenue=("revenue","sum"))
stl = STL(daily["revenue"], period=7, robust=True).fit()
daily["residual"] = stl.resid
daily["ts_anomaly"] = (abs(daily["residual"]) > 3 * daily["residual"].std()).astype(int)
daily.to_parquet("data/gold/daily_ts_anomalies.parquet", index=False)

with open("runs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(" Outputs created:")
print(" - data/gold/campaign_daily_anomalies.parquet")
print(" - data/gold/daily_ts_anomalies.parquet")
print(" - runs/metrics.json")
print("Metrics:", metrics)
