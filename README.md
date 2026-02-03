# Intelligent Demand Sensing & Autonomous Inventory Planning (IDS)

An enterprise-grade, AI-driven system for **daily demand forecasting and autonomous inventory planning**, designed to translate historical sales data, seasonality effects, and GenAI-powered what-if analysis into **actionable supply chain decisions**.

This repository is intentionally structured so that **any AI copilot or human engineer** can:

* Understand the business problem
* Understand the system design
* Implement each module end-to-end
* Execute the full pipeline
* Reproduce results and insights

---

## 1. Business Problem

Retail and supply chain teams often struggle with inaccurate demand forecasts, especially during seasonal peaks and festival periods. Traditional planning approaches are largely reactive, resulting in:

* Stockouts during high-demand periods
* Overstock and increased holding costs during low-demand periods
* Poor visibility into future demand scenarios
* Limited decision support for planners

The core challenge is to **sense demand early**, forecast it accurately at a daily level, and convert those forecasts into **clear, operational inventory actions**.

---

## 2. Solution Overview

The Intelligent Demand Sensing (IDS) system solves this problem by combining classical machine learning, time-series forecasting, and Generative AI into a single, cohesive pipeline.

High-level flow:

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

The system produces:

* Daily demand forecasts
* Demand classification (Low / Medium / High)
* Safety stock recommendations (95% CSL)
* Natural-language what-if scenario analysis

---

## 3. System Architecture

The project follows a **modular, production-oriented architecture**, where each stage of the pipeline is isolated, testable, and configurable.

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
Visualization & Business Insights
```

Key architectural principles:

* Config-driven behavior (no hardcoding)
* Clear separation of data, code, and outputs
* Step-wise pipeline execution
* Enterprise-readiness and auditability

---

## 4. Repository Structure (End-State)

```
intelligent-demand-sensing/
│
├── config/
│   └── config.yaml              # Central configuration for the entire system
│
├── data/
│   ├── raw/                     # Immutable customer-provided raw data
│   ├── calendar/                # Holiday and festival calendars
│   └── processed/               # Cleaned and feature-engineered datasets
│
├── outputs/
│   ├── forecasts/               # Final demand forecasts
│   ├── plots/                   # Visualizations
│   └── reports/                 # Model evaluation & business insights
│
├── logs/                         # Execution logs
│
├── src/
│   ├── ingestion01/             # Step 1: Data ingestion & preprocessing
│   │   └── preprocessing.py
│   │
│   ├── features02/              # Step 2: Feature engineering
│   │   └── feature_engineering.py
│   │
│   ├── models03/                # Step 3: Forecasting models
│   │   ├── model_trainer.py     # Model training logic (no tuning)
│   │   └── model_registry.py    # Model saving & loading
│   │
│   ├── inventory04/             # Step 4: Inventory planning logic
│   │   └── inventory_planning.py
│   │
│   ├── llm05/                   # Step 5: GenAI what-if analysis
│   │   └── llm_agent.py
│   │
│   ├── pipelines/               # End-to-end orchestration
│   │   └── run_pipeline.py
│   │
│   └── utils/                   # Shared utilities
│       ├── config_loader.py     # Safe YAML config loader
│       └── logger.py            # Centralized logging
│
├── tests/                        # Unit tests
│
├── main.py                       # Single project entry point
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .gitignore
```

---

## 5. Data Description

* **Time Period**: December 2022 – December 2024
* **Granularity**: Daily
* **Key Metrics**:

  * Sales Open
  * Sales Closed
  * Total Sales

### External Enrichment

* Holiday and festival calendar
* Day-of-week indicators
* Seasonal patterns

The dataset is designed to reflect realistic retail demand behavior, including spikes during festivals and variability across weeks and months.

---

## 6. Feature Engineering

Feature engineering is fully controlled via `config.yaml` and includes:

* Lag features (1, 7, 14, 30 days)
* Rolling statistics (mean, trends)
* Calendar joins (holidays, festivals)
* Demand bucket classification (Low / Medium / High)

This layer converts raw sales data into **model-ready signals**.

---

## 7. Modeling Approach

To ensure robustness and accuracy, multiple models are trained and compared:

* **Regression Models**: Linear Regression, Random Forest
* **Gradient Boosting**: XGBoost, LightGBM
* **Time-Series Models**: Prophet

A **stacked ensemble** combines individual model predictions. Models are evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)

Forecast outputs are then classified into demand categories to support operational planning.

---

## 8. Inventory Planning Logic

Forecasted demand is translated into inventory decisions using:

* Safety stock calculation
* 95% Customer Service Level (CSL)
* Demand variability assumptions
* Lead-time considerations

This layer bridges the gap between **data science outputs** and **real-world supply chain execution**.

---

## 9. GenAI What-If Analysis Module

The IDS system includes a lightweight, guardrailed LLM agent that allows planners to ask natural-language questions such as:

* "What if demand increases by 20% during Diwali?"
* "What if supplier lead time doubles?"

The agent:

* Uses prompt-based reasoning
* Applies rule-based and regex fallbacks
* Produces explainable, safe outputs suitable for enterprise usage

---

## 10. How to Run the Project

### Environment Setup

1. Create and activate the Conda environment:

```
conda create -n ids_env python=3.11 -y
conda activate ids_env
```

2. Install dependencies:

```
pip install -r requirements.txt
```

### Execution

* Run individual modules from the `src/` directory during development
* Use `main.py` or `pipelines/run_pipeline.py` for end-to-end execution

---

## 11. Results & Insights

This section will contain:

* Model evaluation metrics
* Forecast visualizations
* Business insights and recommendations

(Added after training and validation)

---

## 12. Roadmap

* Hyperparameter tuning and cross-validation
* Automated pipelines and MLOps integration
* Interactive dashboards
* ERP / inventory system integration
* Real-time demand sensing

---

## 13. Disclaimer

This project is for learning and demonstration purposes and uses simulated or anonymized data. It is not intended for direct production deployment without further validation.
