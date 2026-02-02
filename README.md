# Intelligent Demand Sensing & Autonomous Inventory Planning (IDS)

AI-driven system for daily demand forecasting and inventory optimization that translates historical sales data, seasonality, and GenAI-powered what-if analysis into actionable inventory decisions.

---

## 1. Business Problem

Retail and supply chain teams often struggle with inaccurate demand forecasts, especially during seasonal peaks and festivals. Traditional planning approaches are reactive, leading to:

* Stockouts during high-demand periods
* Overstock and increased holding costs during low-demand periods
* Poor visibility into future scenarios and decision impact

The challenge is to **sense demand early**, forecast it accurately, and translate forecasts into **actionable inventory decisions**.

---

## 2. Solution Overview

The IDS system addresses this problem by combining classical machine learning, time-series forecasting, and Generative AI:

```
Historical Sales Data
        ↓
Data Ingestion & Preprocessing
        ↓
Feature Engineering (seasonality, holidays, trends)
        ↓
Forecasting Models (ML + Time Series)
        ↓
Ensemble Forecast & Demand Classification
        ↓
Safety Stock & Inventory Planning
        ↓
LLM-powered What-If Analysis
        ↓
Business Insights & Decisions
```

The system produces **daily demand forecasts**, classifies demand levels, recommends safety stock, and allows planners to simulate business scenarios using natural language.

---

## 3. System Architecture

The project follows a modular, production-oriented architecture:

```
Raw Data
   ↓
Ingestion Layer
   ↓
Preprocessing & Feature Engineering
   ↓
Forecasting Models (LR, RF, XGB, LGB, Prophet)
   ↓
Ensemble & Demand Classification
   ↓
Inventory Planning (Safety Stock @ 95% CSL)
   ↓
LLM What-If Agent (Guardrailed)
   ↓
Visualization & Insights
```

Each layer is isolated and testable, enabling scalability and future MLOps integration.

---

## 4. Data Description

* **Time Period**: December 2022 – December 2024
* **Granularity**: Daily sales
* **Key Metrics**:

  * Sales Open
  * Sales Closed
  * Total Sales
* **External Enrichment**:

  * Festival and holiday calendar
  * Day-of-week and seasonality indicators

The dataset is designed to reflect real-world retail demand patterns.

---

## 5. Modeling Approach

To ensure robustness and accuracy, multiple models are trained and compared:

* **Regression Models**: Linear Regression, Random Forest
* **Gradient Boosting**: XGBoost, LightGBM
* **Time Series**: Prophet

A **stacked ensemble** combines the strengths of individual models and is evaluated using standard regression metrics such as MAE, RMSE, and MAPE. Forecast outputs are then classified into **Low / Medium / High demand buckets** to support operational decisions.

---

## 6. Inventory Planning Logic

The forecasted demand is translated into inventory actions using:

* Safety stock estimation
* 95% customer service level (CSL)
* Demand variability and lead-time assumptions

This bridges the gap between data science outputs and supply chain execution.

---

## 7. GenAI What-If Analysis Module

A lightweight LLM-based agent enables planners to ask questions such as:

* "What if demand increases by 20% during Diwali?"
* "What if supplier lead time doubles?"

The agent uses prompt-based logic with rule-based fallbacks to ensure **safe, explainable outputs**.

---

## 8. Project Structure

```
intelligent-demand-sensing/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── forecasting/
│   ├── ensemble/
│   ├── inventory_planning/
│   ├── llm_agent/
│   └── visualization/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9. How to Run the Project

1. Create and activate the Conda environment:

```bash
conda create -n ids_env python=3.11 -y
conda activate ids_env
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run individual modules from the `src/` directory as needed.

---

## 10. Results & Insights

Model evaluation metrics, visualizations, and insights will be added after training and validation.

---

## 11. Roadmap

* Hyperparameter tuning and cross-validation
* Automated pipelines and MLOps integration
* Interactive dashboards
* ERP / inventory system integration
* Real-time demand sensing

---

## 12. Disclaimer

This project is for learning and demonstration purposes and uses simulated or anonymized data.
