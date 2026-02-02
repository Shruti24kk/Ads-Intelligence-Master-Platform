# Ads Intelligence Master Platform

**PySpark • Databricks • FastAPI • Anomaly Detection • LLM Summarization • CI/CD**

An end-to-end **Ads Intelligence platform** for analyzing large-scale advertising data, detecting anomalous campaign behavior, and generating **human-readable insights using LLMs**.

This project is designed as a **production-oriented, modular system** that mirrors how modern ads and analytics platforms are built: scalable data processing, service-based APIs, and AI components that assist human decision-making rather than replace deterministic systems.

---

## 🔍 Problem Statement

Advertising platforms generate **millions of events** across impressions, clicks, conversions, and spend.  
At scale, it becomes difficult to:

- Detect sudden performance drops or spikes across campaigns
- Identify hidden anomalies in noisy, high-volume metrics
- Translate raw metrics into actionable insights for analysts and stakeholders

This platform addresses those challenges by combining:

- Distributed data processing for large-scale event analytics  
- ML-based anomaly detection for campaign health monitoring  
- LLM-driven summarization to convert anomalies into interpretable explanations  
- Service-oriented APIs to support integration with downstream tools

---

## 🏗️ System Overview

The platform is organized into **three core layers**:

1. **Data Processing Layer (Spark / Databricks)**  
   - Processes large-scale ad-event data using PySpark  
   - Aggregates raw events into campaign-level metrics  

2. **Service Layer (FastAPI Microservices)**  
   - Exposes anomaly detection and summarization as REST APIs  
   - Designed to support experimentation and downstream integration  

3. **AI & Intelligence Layer**  
   - ML-based anomaly detection models  
   - LLM-based summarization to explain anomalous behavior in natural language  

---

## ✨ Key Features

### 📊 Scalable Data Processing
- PySpark-based ETL pipeline for ad-event data  
- Designed to scale to **1.5M+ events**  
- Compatible with **Databricks runtime**  

### 🚨 Anomaly Detection
- Campaign-level anomaly detection using ML techniques (e.g., Isolation Forest)  
- Identifies abnormal spikes, drops, and unusual metric patterns  
- Can be invoked via batch jobs or REST APIs  

### 🤖 LLM-Based Summarization
- Modular LLM component for summarizing campaign performance  
- Converts numerical metrics and anomalies into human-readable insights  
- Designed for analyst-assist workflows (not autonomous decision-making)  

### 🌐 Microservice Architecture
- FastAPI-based services for:
  - Anomaly detection  
  - Campaign summarization  
- Clear separation between data processing and inference layers  

### 🔍 Observability & CI
- Structured logging for pipeline and service execution  
- GitHub Actions-based CI for validation and code health checks  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Distributed Processing:** PySpark  
- **Analytics / ML:** Scikit-learn  
- **APIs:** FastAPI  
- **LLM Integration:** OpenAI / prompt-based workflows  
- **Platform:** Databricks (notebook + Spark runtime)  
- **CI/CD:** GitHub Actions  

---

## 📁 Project Structure

