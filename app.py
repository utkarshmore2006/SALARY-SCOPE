import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Salary Scope", layout="centered")

# Load model and encoders
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Load dataset for interactive charts
df = pd.read_csv("salary_scope_analytics.csv")

# Page title
st.title("💼 SALARY-SCOPE: Job Salary Predictor")
st.markdown("Get an estimated salary in USD based on your job profile inputs.")

# Friendly Labels for Dropdowns
experience_labels = {
    'EN': 'Entry-level',
    'MI': 'Mid-level',
    'SE': 'Senior-level',
    'EX': 'Executive-level'
}
employment_labels = {
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}
company_size_labels = {
    'S': 'Small',
    'M': 'Medium',
    'L': 'Large'
}

# Reverse maps
experience_reverse = {v: k for k, v in experience_labels.items()}
employment_reverse = {v: k for k, v in employment_labels.items()}
company_size_reverse = {v: k for k, v in company_size_labels.items()}

# User Inputs
st.header("📝 Enter Your Job Details")

experience_input = st.selectbox("Experience Level", list(experience_labels.values()))
employment_input = st.selectbox("Employment Type", list(employment_labels.values()))
job_input = st.selectbox("Job Title", encoders['job_title'].classes_)
company_size_input = st.selectbox("Company Size", list(company_size_labels.values()))
remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 50)

# Encode inputs
input_data = [
    encoders['experience_level'].transform([experience_reverse[experience_input]])[0],
    encoders['employment_type'].transform([employment_reverse[employment_input]])[0],
    encoders['job_title'].transform([job_input])[0],
    encoders['company_size'].transform([company_size_reverse[company_size_input]])[0],
    remote_ratio
]

input_array = np.array(input_data).reshape(1, -1)

# Salary Prediction
if st.button("🔍 Predict Salary"):
    prediction = model.predict(input_array)[0]
    st.success(f"💰 Estimated Salary: **${int(prediction):,} USD**")

st.markdown("---")

# 📊 Interactive Charts Section (Replacing Screenshots)
st.header("📈 Interactive Insights Dashboard")

# Top 10 Highest Paying Jobs
st.subheader("💰 Top 10 Highest Paying Job Titles")
top_jobs = df.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False).head(10)
fig1 = px.bar(top_jobs, x=top_jobs.values, y=top_jobs.index, orientation='h',
              color=top_jobs.values, color_continuous_scale='viridis',
              labels={"x": "Average Salary (USD)", "y": "Job Title"})
st.plotly_chart(fig1, use_container_width=True)

# Salary Distribution
st.subheader("📊 Salary Distribution (Histogram)")
fig2 = px.histogram(df, x='salary_in_usd', nbins=30, color_discrete_sequence=['#4C78A8'])
st.plotly_chart(fig2, use_container_width=True)

# Salary by Experience Level
st.subheader("🎓 Salary by Experience Level")
fig3 = px.box(df, x='experience_level', y='salary_in_usd', color='experience_level')
st.plotly_chart(fig3, use_container_width=True)

# Average Salary by Company Size
st.subheader("🏢 Average Salary by Company Size")
fig4 = px.bar(df, x='company_size', y='salary_in_usd', color='company_size',
              title='Company Size vs Salary', barmode='group')
st.plotly_chart(fig4, use_container_width=True)

# Salary vs Remote Ratio (Scatter)
st.subheader("🌐 Salary vs Remote Work Ratio")
fig5 = px.scatter(df, x='remote_ratio', y='salary_in_usd', color='experience_level',
                  hover_data=['job_title'], title="Remote Ratio vs Salary")
st.plotly_chart(fig5, use_container_width=True)

# New: Pie Chart - Distribution of Job Titles (Top 5)
st.subheader("🧩 Top 5 Job Roles by Frequency (Pie Chart)")
top_jobs_count = df['job_title'].value_counts().head(5)
fig6 = px.pie(values=top_jobs_count.values, names=top_jobs_count.index,
              title="Most Common Job Titles", color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# 📥 Power BI File Downloads
st.header("📊 Download Full Power BI Dashboard")

try:
    with open("Salary_Scope_Dashboard.pbix", "rb") as file:
        st.download_button("⬇️ Download Power BI File (.pbix)", file, file_name="Salary_Scope_Dashboard.pbix")

    with open("Salary_Scope_Report.pdf", "rb") as file:
        st.download_button("⬇️ Download Dashboard Report (.pdf)", file, file_name="Salary_Scope_Report.pdf")
except FileNotFoundError:
    st.warning("Power BI files not found. Make sure they are placed in the same folder as `app.py`.")

# Footer
st.markdown("---")
st.caption("📌 Project by UTKARSH JITENDRA MORE | Built with Python, Streamlit, and Plotly (Power BI Style)")


