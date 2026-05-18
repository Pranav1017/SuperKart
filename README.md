# SuperKart Sales Forecasting & Deployment

End-to-end machine learning solution to forecast quarterly product-level sales revenue for SuperKart, a multi-city retail chain. The project covers the full ML lifecycle — from exploratory data analysis to a deployed REST API and interactive frontend.

**Live Demo → [SuperKart Sales Predictor](https://huggingface.co/spaces/Ramasita0711/SuperKartUI)**  
**API → [Backend on Hugging Face](https://huggingface.co/spaces/Ramasita0711/SuperKartPredictorBackend)**

---

## Business Context

SuperKart operates supermarkets and food marts across Tier 1, 2, and 3 cities, selling thousands of product-store combinations. Accurate sales forecasting helps the business:

- Optimize inventory and procurement planning
- Guide regional sales strategies
- Reduce pipeline risk and improve supply chain decisions

The goal was not just to build a model, but to deploy it as a production-ready forecasting system accessible via API.

---

## Dataset

| Property | Detail |
|---|---|
| Records | 8,523 product-store combinations |
| Features | 11 input features (product + store attributes) |
| Target | `Product_Store_Sales_Total` — total revenue per product per store |
| Sales range | $33 – $8,000 (mean ~$3,464) |

**Key features:**
- `Product_MRP` — maximum retail price
- `Product_Allocated_Area` — ratio of display area allocated to the product
- `Store_Type` — Supermarket Type 1/2, Departmental Store, Food Mart
- `Store_Location_City_Type` — Tier 1, 2, or 3
- `Product_Sugar_Content` — Low Sugar, Regular, No Sugar
- `Store_Age` *(engineered)* — years since store establishment
- `Grouped_Product_Type` *(engineered)* — 16 product types consolidated into 4 categories

---

## Project Pipeline

```
Data Loading → Cleaning → EDA → Feature Engineering → Modeling → Tuning → Deployment
```

1. **Data Cleaning** — Standardized inconsistent category labels, imputed missing values for `Product_Weight` and `Store_Size`
2. **EDA** — Univariate and bivariate analysis; distribution plots, correlation heatmaps, sales breakdown by store type and city tier
3. **Feature Engineering** — Created `Store_Age` from establishment year; grouped 16 product types into 4 broader categories
4. **Modeling** — Trained 6 regression models inside sklearn Pipelines (preprocessing + model in one object)
5. **Hyperparameter Tuning** — GridSearchCV on XGBoost
6. **Deployment** — Flask REST API + Streamlit UI, both containerized with Docker and deployed to Hugging Face Spaces

---

## Models Compared

| Model | Train R² | Notes |
|---|---|---|
| Decision Tree | 1.00 | Severe overfitting |
| Bagging Regressor | ~0.96 | Reduces variance |
| Random Forest | ~0.97 | Strong baseline |
| AdaBoost | ~0.55 | Underfits |
| Gradient Boosting | ~0.75 | Good generalization |
| **XGBoost** ✦ | **~0.89** | **Best bias-variance tradeoff — selected for deployment** |

XGBoost was selected based on lowest RMSE and best generalization to the test set.

---

## Deployment Architecture

```
Streamlit UI  →  Flask REST API (/v1/sales)  →  XGBoost Pipeline  →  JSON Response
  (HF Space)        (HF Docker Space)              (joblib model)
```

- **Backend:** Flask app served via Gunicorn (4 workers), containerized with Docker
- **Frontend:** Streamlit app containerized with Docker, calls the Flask API
- **Model storage:** Serialized pipeline stored on Hugging Face Hub via `joblib`

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | Python 3.9, Pandas 2.2, NumPy 2.0 |
| Visualization | Matplotlib 3.10, Seaborn 0.13 |
| Modeling | Scikit-learn 1.6, XGBoost 2.1, joblib |
| API | Flask 2.2, Gunicorn |
| Frontend | Streamlit 1.43 |
| Infrastructure | Docker, Hugging Face Spaces, Hugging Face Hub |

---

## Key Business Insights

- **Supermarket Type 2** outlets contribute disproportionately to sales — prioritize inventory and marketing here
- **Tier 2 cities** underperform relative to their size — opportunity for targeted promotions
- **Food Items** dominate revenue — diverse, well-stocked food inventory directly protects top-line sales
- **Higher MRP products** correlate with stronger sales — strategic placement of premium products can lift revenue
- **Product Allocated Area** is a meaningful predictor — display space influences sales performance across store types

---

## Repository Structure

```
superkart-sales-forecasting/
│
├── RamaSita_Full_Code_SuperKart_Model_Deployment_Notebook.ipynb   # Full notebook
├── RamaSita_Full_Code_SuperKart_Model_Deployment_Notebook.html    # Rendered notebook
├── SuperKart.csv                                                   # Dataset
└── README.md
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/superkart-sales-forecasting
cd superkart-sales-forecasting

# Install dependencies
pip install numpy==2.0.2 pandas==2.2.2 scikit-learn==1.6.1 xgboost==2.1.4 \
            matplotlib==3.10.0 seaborn==0.13.2 joblib==1.4.2

# Open the notebook
jupyter notebook RamaSita_Full_Code_SuperKart_Model_Deployment_Notebook.ipynb
```

---

## Program

Post Graduate Program in Artificial Intelligence and Machine Learning  
**The University of Texas at Austin** — McCombs School of Business  
*In collaboration with Great Learning*
