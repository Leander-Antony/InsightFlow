# eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def basic_stats(df):
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe(include='all'))

    st.subheader("🧼 Missing Values")
    st.dataframe(df.isnull().sum())

def plot_correlation_matrix(df):
    st.subheader("🔗 Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()

    st.write(corr)

    if len(corr.columns) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("Not enough numerical features for correlation heatmap.")

def full_eda(df):
    basic_stats(df)
    plot_correlation_matrix(df)
