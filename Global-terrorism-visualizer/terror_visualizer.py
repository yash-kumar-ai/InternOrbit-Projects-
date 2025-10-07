import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# Load Data
# -------------------------------
st.set_page_config(page_title="Global Terrorism Visualizer", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("globalterrorismdb.csv", encoding="ISO-8859-1", low_memory=False)
    return df

df = load_data()

st.title("🌍 Global Terrorism Analysis & Visualization")

# -------------------------------
# Filters
# -------------------------------
years = sorted(df["iyear"].unique())
countries = sorted(df["country_txt"].dropna().unique())

col1, col2 = st.columns(2)

with col1:
    selected_year = st.slider("Select Year", int(min(years)), int(max(years)), (2000, 2015))
with col2:
    selected_country = st.multiselect("Select Countries", countries, default=["India", "Iraq", "Pakistan"])

filtered_df = df[(df["iyear"] >= selected_year[0]) & (df["iyear"] <= selected_year[1])]
if selected_country:
    filtered_df = filtered_df[filtered_df["country_txt"].isin(selected_country)]

# -------------------------------
# Plots
# -------------------------------
st.subheader("📊 Number of Attacks Per Year")
yearly_attacks = filtered_df.groupby("iyear").size().reset_index(name="attacks")
fig1 = px.line(yearly_attacks, x="iyear", y="attacks", markers=True, title="Yearly Terrorist Attacks")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📍 Attacks on World Map")
fig2 = px.scatter_geo(
    filtered_df,
    lat="latitude",
    lon="longitude",
    hover_name="country_txt",
    scope="world",
    color="region_txt",
    size_max=10,
    opacity=0.7,
    title="Geographical Distribution of Terrorist Attacks"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🎯 Attack Types Distribution")
attack_counts = filtered_df["attacktype1_txt"].value_counts().reset_index()
attack_counts.columns = ["Attack Type", "Count"]
fig3 = px.bar(attack_counts, x="Attack Type", y="Count", title="Attack Types")
st.plotly_chart(fig3, use_container_width=True)

st.success("Visualization ready ✅ Use the sidebar to filter by year and country!")
