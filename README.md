# Ads-Intelligence-Master-Platform
# Ads Intelligence Master Platform
Anomaly Detection • LLM Evaluation • Causal Inference • Bandits • Spark • Azure ML

## Overview
This project simulates a production-grade advertising intelligence platform similar to systems used in Microsoft Advertising.

It integrates large-scale data processing, anomaly detection, LLM evaluation, causal inference, and reinforcement learning into a single end-to-end system.

## Key Capabilities
- Large-scale anomaly detection on clickstream data
- Forecasting + time-series anomaly surfacing
- LLM summarization with prompt optimization and evaluation
- Causal inference and uplift modeling
- Contextual bandits for decision optimization
- Spark / Databricks lakehouse pipelines
- Azure ML training and deployment templates

## Architecture
Raw Events → Bronze → Silver → Gold → ML Systems → Azure ML / Databricks

## How to Run (Local)
```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/run_pipeline.py
