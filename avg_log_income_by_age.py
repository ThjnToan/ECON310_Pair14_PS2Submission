import pandas as pd
import glob
import os
import numpy as np

folder = r'VHLSS 2008 Data'
household_folder = os.path.join(folder, 'male_head_households')

# Collect all household files
target_files = glob.glob(os.path.join(household_folder, 'household_*.csv'))

# List to store age and log_income
data = []
for file in target_files:
    df = pd.read_csv(file)
    if 'age' in df.columns and 'log_income' in df.columns:
        # Only keep rows with valid age and log_income
        df = df[['age', 'log_income']].dropna()
        data.append(df)

# Combine all data
if data:
    all_data = pd.concat(data, ignore_index=True)
    # Convert age to numeric (if not already)
    all_data['age'] = pd.to_numeric(all_data['age'], errors='coerce')
    all_data = all_data.dropna(subset=['age', 'log_income'])
    all_data['age'] = all_data['age'].astype(int)
    # Group by age and calculate average log_income
    avg_log_income = all_data.groupby('age')['log_income'].mean().reset_index()
    avg_log_income = avg_log_income.rename(columns={'log_income': 'avg_log_income'})
    # Add avg_income column
    avg_log_income['avg_income'] = avg_log_income['avg_log_income'].apply(lambda x: np.exp(x))
    # Save to CSV
    avg_log_income.to_csv(os.path.join(folder, 'avg_log_income_by_age.csv'), index=False)
    print('Saved avg_log_income_by_age.csv')
else:
    print('No valid data found in household files.')
