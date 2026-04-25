# 🌾 Predicting Crop Production Based on Agricultural Data

A machine learning project that forecasts crop production (in tons) using FAOSTAT agricultural data across 200 countries and 257 crop types (2019–2023).

---

## 📌 Problem Statement

Agriculture is a key contributor to the economy, and accurately predicting crop production is essential for improving planning and decision-making. This project develops a regression model that forecasts crop production (in tons) based on agricultural factors such as area harvested (ha), yield (kg/ha), and year, for various crops grown across global regions.

---

## 🗂️ Project Structure

```
├── app.py                          # Streamlit application (5-page dashboard)
├── Crop_Production_Full_Analysis.ipynb  # EDA, model training & evaluation notebook
├── cleansed_crop_data.csv          # Cleaned FAOSTAT dataset (78,364 rows)
├── model_results.json              # Model comparison metrics
├── eda_stats.json                  # Precomputed EDA statistics
└── README.md
```

---

## 📊 Dataset

**Source:** FAOSTAT (UN Food and Agriculture Organization)

| Column | Description |
|---|---|
| `Area` | Country / region name |
| `Item` | Crop / product name |
| `Item Code (CPC)` | FAO classification code for the crop |
| `Year` | Calendar year (2019–2023) |
| `Area_Harvested_in_Hectares` | Land area harvested (ha) |
| `Yield_Value in kg/ha` | Yield per hectare |
| `Production in Hectares` | Total production in tons (target variable) |
| Livestock columns | Producing animals, laying, carcass weight, milk animals |

---

## 🔧 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crop-production-prediction.git
cd crop-production-prediction

# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly joblib

# Run the Streamlit app
streamlit run app.py
```

> **Note:** The model is trained automatically on first launch from the CSV and cached locally as `model.pkl`.

---

## 🤖 Models Evaluated

| Model | R² Score | MAE (tons) | RMSE (tons) |
|---|---|---|---|
| Linear Regression | 0.6721 | 909,301 | 7,586,839 |
| Decision Tree | 0.9312 | 202,462 | 3,476,140 |
| **Random Forest** ✅ | **0.9722** | **174,236** | **2,208,656** |
| Gradient Boosting | 0.9743 | 391,699 | 2,125,591 |

**Best model deployed:** Random Forest (best MAE — most reliable for real-world use)

---

## 📱 Streamlit App Pages

1. **🏠 Overview** — KPIs, global production by year, top countries and crops
2. **📊 EDA & Insights** — Distributions, correlation heatmap, outlier analysis, data preview
3. **📈 Trends & Comparisons** — Filtered trends, cross-country choropleth map, crop comparison
4. **🤖 Model Performance** — Side-by-side metrics and charts for all 4 models
5. **🔮 Predict Production** — Interactive prediction form using the trained Random Forest model

---

## 💡 Key Insights

- **Area Harvested** is the strongest predictor of production (~0.85 correlation)
- **China, India, and the USA** dominate global agricultural output
- Tree-based models significantly outperform Linear Regression due to non-linear relationships in the data
- Production values are highly skewed — a small number of country-crop combinations account for the majority of global output

---

## 🏢 Business Use Cases

- 🏛 **Food Security & Planning** — Help governments anticipate supply shortfalls
- 📋 **Agricultural Policy** — Inform subsidy and insurance programs
- 🚚 **Supply Chain Optimization** — Plan storage and logistics ahead of harvest
- 💰 **Market Price Forecasting** — Enable better trading decisions for farmers
- 🌱 **Precision Farming** — Guide crop selection based on regional productivity

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `Plotly` · `Streamlit` · `Joblib`

---

## 📁 Domain

**Agriculture** | Data Cleaning · EDA · Machine Learning (Regression) · Data Visualization · Streamlit
