import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =======================================
#      D3. Data Preparation Steps
# =======================================

# Step 1: Load Data
df = pd.read_csv('data/churn_clean.csv')

# Step 2: Select Relevant Variables
selected_columns = [
    'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Income',
    'Outage_sec_perweek', 'Contract', 'InternetService',
    'TechSupport', 'DeviceProtection', 'StreamingTV',
    'PaymentMethod', 'PaperlessBilling'
]

df = df[selected_columns]

# Step 3: Handle Missing Values
print(df.isnull().sum())
print(df.size)

# Handle missing InternetService
df['InternetService'] = df['InternetService'].fillna('None')

# Check if any other missing values exist
print(df.isnull().sum())
print(df.size)

# Step 4: Convert Categorical Variables to String Type
categorical_cols = [
    'Contract', 'InternetService', 'TechSupport',
    'DeviceProtection', 'StreamingTV',
    'PaymentMethod', 'PaperlessBilling'
]

for col in categorical_cols:
    df[col] = df[col].astype(str)

# Step 5: Normalize Continuous Variables
continuous_cols = [
    'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year',
    'Income', 'Outage_sec_perweek'
]

scaler = MinMaxScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# Step 6: Save Cleaned Dataset
df.to_csv('data/cleaned_churn_data.csv', index=False)