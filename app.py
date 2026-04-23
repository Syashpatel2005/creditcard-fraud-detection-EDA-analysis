import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

pipeline = joblib.load("fraud_model.pkl")

xgb_model = pipeline.named_steps['model']

feature_names = xgb_model.feature_names_in_

importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

st.set_page_config(layout="wide")

st.title("📊 Credit Card Fraud Detection — Insights Dashboard")

# Load dataset
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1i3spxjoDVPHpeWPfrtfV49jGb2ONKa6I"
    gdown.download(url, "creditcard.csv", quiet=False)
    pd.read_csv("creditcard.csv")
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Overview",
    "Class Distribution",
    "Time Analysis",
    "Amount Analysis",
    "Correlation Heatmap",
    "Final Insights",
    "Feature Importance"
])

# ---------------------------
# 📌 Overview
# ---------------------------
if section == "Overview":
    st.subheader("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", df['Class'].sum())
    col3.metric("Fraud %", f"{df['Class'].mean()*100:.4f}%")

    st.warning("⚠️ Dataset is highly imbalanced")

# ---------------------------
# 📊 Class Distribution
# ---------------------------
elif section == "Class Distribution":
    st.subheader("📊 Class Distribution")

    fig = px.histogram(df, x='Class', color='Class',
                       title="Fraud vs Legit Transactions",
                       log_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### 🔍 Insight:
- Fraud cases are extremely rare (~0.17%)
- Dataset is highly imbalanced

### 💡 Meaning:
- Accuracy is NOT reliable
- Model must focus on Recall & Precision

### ✅ Conclusion:
Use **AUPRC instead of Accuracy**
""")

# ---------------------------
# ⏱ Time Analysis
# ---------------------------
elif section == "Time Analysis":
    st.subheader("⏱ Transaction Time Analysis")

    class_0 = df[df['Class'] == 0]['Time']
    class_1 = df[df['Class'] == 1]['Time']

    fig = ff.create_distplot(
        [class_0, class_1],
        ['Legit', 'Fraud'],
        show_hist=False,
        show_rug=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### 🔍 Insight:
- Fraud transactions occur in clusters

### 💡 Meaning:
- Fraudsters act in patterns (not random)

### ✅ Conclusion:
Time is an important feature
""")

# ---------------------------
# 💰 Amount Analysis
# ---------------------------
elif section == "Amount Analysis":
    st.subheader("💰 Transaction Amount Analysis")

    fig = px.box(df, x='Class', y='Amount', color='Class')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### 🔍 Insight:
- Fraud transactions often have unusual amounts

### 💡 Meaning:
- Small test transactions + large fraud spikes

### ✅ Conclusion:
Amount helps detect anomalies
""")

# ---------------------------
# 🔥 Correlation Heatmap
# ---------------------------
elif section == "Correlation Heatmap":
    st.subheader("🔥 Feature Correlation")

    fig = px.imshow(
        df.corr(),
        text_auto=False,  
        aspect="auto",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### 🔍 Insight:
- Features have low correlation

### 💡 Meaning:
- PCA transformed features

### ✅ Conclusion:
Good for machine learning models
""")

# ---------------------------
# 🧠 Final Insights
# ---------------------------
elif section == "Final Insights":
    st.subheader("🧠 Final Insights")

    st.success("""
### 🚀 Key Findings:

1. Dataset is highly imbalanced  
2. Fraud transactions follow patterns  
3. Time and Amount are key features  
4. PCA features reduce noise  
5. Fraud is detectable using ML  

### 🎯 Final Conclusion:
Fraud detection requires advanced models like XGBoost with SMOTE and evaluation using AUPRC.
""")

elif section == "Feature Importance":
    st.subheader("🔥 Feature Importance")

    fig = px.bar(importance.head(15),title = 'Top 15 Feature Importance')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("""
### 🔍 Insight:
- Top features contribute most to fraud detection
- PCA features dominate importance

### 💡 Meaning:
- Model detects hidden patterns (not obvious rules)

### ✅ Conclusion:
- XGBoost effectively identifies fraud signals
""")
