import pandas as pd

# Load your original dataset
df = pd.read_csv("ds_salaries.csv")

# Select relevant columns
df_model = df[['experience_level', 'employment_type', 'job_title', 'company_size', 'remote_ratio', 'salary_in_usd']]

# Save the cleaned dataset
df_model.to_csv("salary_scope_analytics.csv", index=False)

print("✅ File 'salary_scope_analytics.csv' created successfully.")
