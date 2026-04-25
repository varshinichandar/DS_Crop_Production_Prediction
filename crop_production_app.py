import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os

st.set_page_config(page_title="Crop Production Predictor", layout="wide")

st.markdown("""
<style>
    .prediction-box {
        background-color: #2e7d32;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .prediction-value {
        font-size: 2.4rem;
        font-weight: bold;
        color: #b9f6ca;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "crop_production_data.csv"))
    return df

@st.cache_resource
def get_model(_df):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    model_path = os.path.join(BASE_DIR, "model.pkl")
    results_path = os.path.join(BASE_DIR, "model_comparison_results.json")

    if os.path.exists(model_path):
        return joblib.load(model_path)

    feature_cols = [
        'Item Code (CPC)', 'Year',
        'Area_Harvested_in_Hectares', 'Yield_Value in kg/ha',
        'Producing Animals/Slaughtered_Value', 'Laying_Value',
        'Yield/Carcass Weight_Value', 'Milk Animals_Value'
    ]
    X = _df[feature_cols]
    y = _df['Production in Hectares']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training model, please wait..."):
        all_models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        results = {}
        for name, m in all_models.items():
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            results[name] = {
                'R2': round(r2_score(y_test, preds), 4),
                'MSE': round(mean_squared_error(y_test, preds), 2),
                'MAE': round(mean_absolute_error(y_test, preds), 2),
                'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 2)
            }

        rf_model = all_models['Random Forest']
        joblib.dump(rf_model, model_path)
        with open(results_path, 'w') as f:
            json.dump(results, f)

    return rf_model

@st.cache_data
def get_model_results():
    path = os.path.join(BASE_DIR, "model_comparison_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        'Linear Regression':  {'R2': 0.6721, 'MSE': 0, 'MAE': 909301, 'RMSE': 7586839},
        'Decision Tree':      {'R2': 0.9312, 'MSE': 0, 'MAE': 202462, 'RMSE': 3476140},
        'Random Forest':      {'R2': 0.9722, 'MSE': 0, 'MAE': 174236, 'RMSE': 2208656},
        'Gradient Boosting':  {'R2': 0.9743, 'MSE': 0, 'MAE': 391699, 'RMSE': 2125591},
    }

df = load_data()
model = get_model(df)
model_results = get_model_results()

areas = sorted(df["Area"].unique())
items = sorted(df["Item"].unique())
years = sorted(df["Year"].unique().tolist())

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Trends", "Model Comparison", "Predict"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
selected_area = st.sidebar.selectbox("Country", areas, index=areas.index("India") if "India" in areas else 0)
selected_item = st.sidebar.selectbox("Crop", items, index=0)
selected_years = st.sidebar.multiselect("Years", years, default=years)

filtered_df = df[
    (df["Area"] == selected_area) &
    (df["Item"] == selected_item) &
    (df["Year"].isin(selected_years))
]

def format_num(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.2f}"


if page == "Overview":
    st.title("🌾 Crop Production Dashboard")
    st.write("This dashboard explores FAOSTAT crop production data across 200 countries and 257 crop types from 2019 to 2023.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Countries", df["Area"].nunique())
    col3.metric("Crop Types", df["Item"].nunique())
    col4.metric("Total Production", format_num(df["Production in Hectares"].sum()) + " tons")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Production by Year")
        yearly = df.groupby("Year")["Production in Hectares"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Production in Hectares",
                     color_discrete_sequence=["#43a047"],
                     labels={"Production in Hectares": "Production (tons)"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top 10 Countries")
        top_c = df.groupby("Area")["Production in Hectares"].sum().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(top_c, x="Production in Hectares", y="Area", orientation="h",
                      color_discrete_sequence=["#1565c0"],
                      labels={"Production in Hectares": "Production (tons)"})
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top 10 Crops by Production")
        top_i = df.groupby("Item")["Production in Hectares"].sum().sort_values(ascending=False).head(10).reset_index()
        fig3 = px.pie(top_i, names="Item", values="Production in Hectares")
        fig3.update_traces(textposition="inside", textinfo="label+percent")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Area Harvested vs Production")
        sample = df.sample(3000, random_state=42)
        fig4 = px.scatter(sample, x="Area_Harvested_in_Hectares", y="Production in Hectares",
                          color="Year", opacity=0.4,
                          labels={"Area_Harvested_in_Hectares": "Area (ha)",
                                  "Production in Hectares": "Production (tons)"},
                          color_continuous_scale="Greens")
        st.plotly_chart(fig4, use_container_width=True)


elif page == "EDA":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Outliers", "Data"])

    with tab1:
        st.subheader("Feature Distributions")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Production in Hectares", nbins=60, log_y=True,
                               color_discrete_sequence=["#43a047"],
                               title="Production Distribution",
                               labels={"Production in Hectares": "Production (tons)"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x="Yield_Value in kg/ha", nbins=60, log_y=True,
                                color_discrete_sequence=["#1565c0"],
                                title="Yield Distribution",
                                labels={"Yield_Value in kg/ha": "Yield (kg/ha)"})
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.histogram(df, x="Area_Harvested_in_Hectares", nbins=60, log_y=True,
                                color_discrete_sequence=["#f57f17"],
                                title="Area Harvested Distribution")
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            top20 = df["Item"].value_counts().head(20).reset_index()
            top20.columns = ["Item", "Count"]
            fig4 = px.bar(top20, x="Count", y="Item", orientation="h",
                          color_discrete_sequence=["#43a047"],
                          title="Top 20 Most Recorded Crops")
            fig4.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.subheader("Correlation Matrix")
        num_cols = [
            "Area_Harvested_in_Hectares", "Yield_Value in kg/ha",
            "Producing Animals/Slaughtered_Value", "Laying_Value",
            "Yield/Carcass Weight_Value", "Milk Animals_Value", "Production in Hectares"
        ]
        corr = df[num_cols].corr()
        labels = ["Area", "Yield", "ProdAnimals", "Laying", "Carcass", "Milk", "Production"]
        fig = px.imshow(corr.values, x=labels, y=labels,
                        color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                        text_auto=".2f", title="Feature Correlation Heatmap")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.info("Area Harvested has the strongest correlation with Production (0.51). Laying value also shows a notable correlation of 0.58, mainly relevant for egg-producing items.")

    with tab3:
        st.subheader("Outlier Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, y="Production in Hectares", log_y=True,
                         color_discrete_sequence=["#43a047"],
                         title="Production Boxplot")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, y="Yield_Value in kg/ha", log_y=True,
                          color_discrete_sequence=["#1565c0"],
                          title="Yield Boxplot")
            st.plotly_chart(fig2, use_container_width=True)

        q99 = df["Production in Hectares"].quantile(0.99)
        outliers = df[df["Production in Hectares"] > q99][
            ["Area", "Item", "Year", "Production in Hectares"]
        ].sort_values("Production in Hectares", ascending=False).head(15)
        st.write("Top outlier records (above 99th percentile):")
        st.dataframe(outliers.reset_index(drop=True), use_container_width=True)

    with tab4:
        st.subheader("Dataset Preview")
        st.write(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download Dataset", df.to_csv(index=False), "crop_production_data.csv", mime="text/csv")


elif page == "Trends":
    st.title("Trends & Comparisons")

    tab1, tab2, tab3 = st.tabs(["Filtered View", "Country Comparison", "Crop Comparison"])

    with tab1:
        st.write(f"Showing: **{selected_item}** in **{selected_area}**")
        if filtered_df.empty:
            st.warning("No data found for this combination. Try different filters.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Production (tons)", format_num(filtered_df["Production in Hectares"].mean()))
            c2.metric("Avg Yield (kg/ha)", format_num(filtered_df["Yield_Value in kg/ha"].mean()))
            c3.metric("Avg Area (ha)", format_num(filtered_df["Area_Harvested_in_Hectares"].mean()))

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Production Over Years", "Yield Over Years"))
            fig.add_trace(go.Scatter(x=filtered_df["Year"], y=filtered_df["Production in Hectares"],
                                     mode="lines+markers", name="Production", line=dict(color="#43a047")), row=1, col=1)
            fig.add_trace(go.Scatter(x=filtered_df["Year"], y=filtered_df["Yield_Value in kg/ha"],
                                     mode="lines+markers", name="Yield", line=dict(color="#1565c0")), row=1, col=2)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Top Countries for: {selected_item}")
        n = st.slider("Number of countries", 5, 30, 15)
        crop_data = df[(df["Item"] == selected_item) & (df["Year"].isin(selected_years))]
        top_n = crop_data.groupby("Area")["Production in Hectares"].sum().sort_values(ascending=False).head(n).reset_index()
        fig = px.bar(top_n, x="Area", y="Production in Hectares",
                     color_discrete_sequence=["#43a047"],
                     labels={"Production in Hectares": "Total Production (tons)"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        area_prod = crop_data.groupby("Area")["Production in Hectares"].sum().reset_index()
        fig_map = px.choropleth(area_prod, locations="Area", locationmode="country names",
                                color="Production in Hectares", color_continuous_scale="Greens",
                                title=f"Global Production - {selected_item}")
        st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        st.subheader(f"Crop Comparison in {selected_area}")
        compare = st.multiselect("Select crops", items, default=items[:5])
        if compare:
            comp = df[(df["Area"] == selected_area) & (df["Item"].isin(compare)) & (df["Year"].isin(selected_years))]
            if not comp.empty:
                fig = px.line(comp, x="Year", y="Production in Hectares", color="Item",
                              markers=True, labels={"Production in Hectares": "Production (tons)"})
                st.plotly_chart(fig, use_container_width=True)

                avg_yield = comp.groupby("Item")["Yield_Value in kg/ha"].mean().sort_values(ascending=False).reset_index()
                fig2 = px.bar(avg_yield, x="Item", y="Yield_Value in kg/ha", color="Item",
                              labels={"Yield_Value in kg/ha": "Avg Yield (kg/ha)"},
                              title="Average Yield by Crop")
                fig2.update_layout(showlegend=False, xaxis_tickangle=30)
                st.plotly_chart(fig2, use_container_width=True)


elif page == "Model Comparison":
    st.title("Model Performance")

    res_df = pd.DataFrame(model_results).T.reset_index()
    res_df.columns = ["Model", "R2", "MSE", "MAE", "RMSE"]

    best = res_df.loc[res_df["R2"].idxmax(), "Model"]
    st.success(f"Best performing model: **{best}**")

    st.subheader("Metrics Table")
    st.dataframe(res_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(res_df, x="Model", y="R2", color="Model", title="R2 Score (higher is better)")
        fig.add_hline(y=0.9, line_dash="dash", line_color="red")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(res_df, x="Model", y="RMSE", color="Model", title="RMSE (lower is better)")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.bar(res_df, x="Model", y="MAE", color="Model", title="MAE (lower is better)")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.bar(res_df, x="Model", y="MSE", color="Model", title="MSE (lower is better)")
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.write("Random Forest and Gradient Boosting both achieve an R2 above 0.97. Random Forest was chosen as the final model since it has the lowest MAE, meaning its predictions are closest to actual values on average.")


elif page == "Predict":
    st.title("Predict Crop Production")
    st.write("Enter the crop details below to get a predicted production value in tons.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        selected_crop = st.selectbox("Crop", items, key="predict_crop")
        item_code_lookup = df.groupby("Item")["Item Code (CPC)"].first().to_dict()
        item_code = item_code_lookup.get(selected_crop, 0.0)
        st.caption(f"Item Code: {item_code}")

        year_input = st.slider("Year", 2019, 2030, 2024)

        a, b = st.columns(2)
        with a:
            area_input = st.number_input("Area Harvested (ha)", min_value=0.0, value=10000.0, step=500.0)
        with b:
            yield_input = st.number_input("Yield (kg/ha)", min_value=0.0, value=2000.0, step=100.0)

        st.write("Livestock values (enter 0 if not applicable):")
        c, d = st.columns(2)
        with c:
            animals = st.number_input("Producing Animals / Slaughtered", min_value=0.0, value=0.0)
            laying = st.number_input("Laying", min_value=0.0, value=0.0)
        with d:
            carcass = st.number_input("Carcass Weight", min_value=0.0, value=0.0)
            milk = st.number_input("Milk Animals", min_value=0.0, value=0.0)

        if st.button("Predict", type="primary", use_container_width=True):
            input_data = pd.DataFrame([[
                item_code, year_input, area_input, yield_input,
                animals, laying, carcass, milk
            ]], columns=[
                "Item Code (CPC)", "Year",
                "Area_Harvested_in_Hectares", "Yield_Value in kg/ha",
                "Producing Animals/Slaughtered_Value", "Laying_Value",
                "Yield/Carcass Weight_Value", "Milk Animals_Value"
            ])
            prediction = model.predict(input_data)[0]
            st.markdown(f"""
            <div class="prediction-box">
                <div>Predicted Production for <b>{selected_crop}</b></div>
                <div class="prediction-value">{prediction:,.0f} tons</div>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Similar records from dataset")
            similar = df[df["Item"] == selected_crop].sort_values(
                by="Area_Harvested_in_Hectares",
                key=lambda x: (x - area_input).abs()
            ).head(5)[["Area", "Item", "Year", "Area_Harvested_in_Hectares", "Yield_Value in kg/ha", "Production in Hectares"]]
            st.dataframe(similar.reset_index(drop=True), use_container_width=True)

    with col_right:
        st.subheader("Model Info")
        rf_res = model_results.get("Random Forest", {})
        st.metric("R2 Score", f"{rf_res.get('R2', 'N/A'):.4f}")
        st.metric("MAE", f"{rf_res.get('MAE', 'N/A'):,.0f} tons")
        st.metric("RMSE", f"{rf_res.get('RMSE', 'N/A'):,.0f} tons")
        st.markdown("---")
        st.write("**What each input means:**")
        st.write("- **Area Harvested**: Total land used for the crop in hectares")
        st.write("- **Yield**: How much is produced per hectare (kg/ha)")
        st.write("- **Year**: The year you want to predict for")
        st.write("- **Livestock fields**: Only fill these for animal-based products")
