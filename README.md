# Intelligent Demand Sensing & Autonomous Inventory Planning (IDS) – v1.0

An enterprise-grade, modular, AI-driven system for **daily demand forecasting, autonomous inventory planning, explainable modeling, and GenAI-powered what-if simulation**.

IDS is designed to be:

- Fully config-driven  
- Reproducible  
- Leakage-safe  
- Industrial-grade  
- Audit-ready  
- Modular and extensible  

This repository enables:

- Technical reviewers to audit architecture and modeling rigor  
- Enterprise stakeholders to understand business value  
- Internal engineers to execute and extend the system  

---

# 1. Business Context

Retail and supply chain operations frequently struggle with:

- Stockouts during demand spikes  
- Overstock during low-demand periods  
- Poor visibility into demand variability  
- No structured what-if simulation capability  

Traditional forecasting stops at prediction.

**IDS goes further** — it translates forecasts into inventory decisions and scenario-driven business insights.

---

# 2. What IDS v1.0 Delivers

## Forecast Outputs

- Daily demand forecast  
- Weekly summary (week-end rows only)  
- Monthly summary  
- Monthly growth %  

## Modeling Outputs

- Multi-model training  
- Hyperparameter tuning  
- Stacking ensemble  
- Best model selection (Primary metric: MAPE %)  
- Feature importance export  
- SHAP explainability plots  
- Versioned model artifacts  
- Experiment metadata snapshot  

## Inventory Outputs

- Safety stock (95% CSL)  
- Reorder recommendations  
- Lead-time aware planning  
- Versioned inventory outputs  

## GenAI Outputs

- Guardrailed what-if simulations  
- Stateless and stateful scenarios  
- Scenario comparison  
- Scenario TTL (auto-expiry)  
- Versioned scenario outputs  
- Markdown scenario summaries  

## Visualization Outputs

- Model performance plots  
- Forecast trend plots  
- Inventory visualizations  
- Feature importance charts  
- SHAP summary plots  

---

# 3. High-Level System Flow

