#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# 🟩 1. Load Dataset Block:-
df = pd.read_csv('ds_salaries.csv')
# 🟦 Data Exploration
print("\n🔹 First 5 rows:")
print(df.head()) ## View first 5 records
print("\n🔹 Shape:", df.shape) # Shape of the data (rows, columns)
print("\n🔹 Column names:", df.columns) # # Column names
print("\n🔹 Info:")
print(df.info()) # Data types and nulls
print("\n🔹 Description:")
print(df.describe()) # Statistical summary
print("\n🔹 Missing Values:\n", df.isnull().sum()) # Null value check
print("\n🔹 Duplicates:", df.duplicated().sum()) # Check for duplicate rows




# 🟦 2. Data Visualization Block
# Salary Distribution
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(df['salary_in_usd'], kde=True, color='skyblue')
plt.title("Salary Distribution (in USD)")
plt.xlabel("Salary")
plt.ylabel("Number of Jobs")
plt.show()

# Jobs by Experience Level
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='experience_level', palette='pastel')
plt.title("Number of Jobs by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Count")
plt.show()

# Average Salary by Company Size
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='company_size', y='salary_in_usd', estimator='mean', palette='Blues')
plt.title("Average Salary by Company Size")
plt.xlabel("Company Size")
plt.ylabel("Average Salary (USD)")
plt.show()

# Salary by Remote Work Ratio
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='remote_ratio', y='salary_in_usd')
plt.title("Salary by Remote Work Ratio")
plt.xlabel("Remote Ratio")
plt.ylabel("Salary (USD)")
plt.show()

# Top 10 Highest Paying Jobs
top_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='magma')
plt.title("Top 10 Highest Paying Job Titles")
plt.xlabel("Average Salary (USD)")
plt.ylabel("Job Title")
plt.tight_layout()
plt.show()

#Salary Distribution by Experience Level
plt.figure(figsize=(8, 6))
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, palette='coolwarm')
plt.title("Salary Distribution by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.tight_layout()
plt.show()




# 🟦 3. Data Preprocessing Block:-

# Select relevant columns
df_model = df[['experience_level', 'employment_type', 'job_title', 'company_size', 'remote_ratio', 'salary_in_usd']]

# Encode categorical features
df_encoded = df_model.copy()
label_encoders = {}

for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le

# Split features and target
X = df_encoded.drop('salary_in_usd', axis=1)
y = df_encoded['salary_in_usd']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 🟦 4. Model Training & Evaluation Block:-
# ✅ Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n🔸 Random Forest Results:")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R² Score:", r2_score(y_test, rf_pred))

# ✅ Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n🔹 Linear Regression Results:")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R² Score:", r2_score(y_test, lr_pred))

# ✅ Model Comparison Table
results = {
    'Model': ['Random Forest', 'Linear Regression'],
    'MAE': [mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, lr_pred)],
    'MSE': [mean_squared_error(y_test, rf_pred), mean_squared_error(y_test, lr_pred)],
    'R² Score': [r2_score(y_test, rf_pred), r2_score(y_test, lr_pred)]
}
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:\n")
print(results_df)



# 🟦 5. Save Model & Encoders
with open('salary_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

with open('encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)



# Save processed dataset to CSV for Power BI
df_model.to_csv("salary_scope_data.csv", index=False)

