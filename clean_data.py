import pandas as pd
import glob
import os
import numpy as np

folder = r'VHLSS 2008 Data'
key_cols = ["tinh", "huyen", "xa", "diaban", "hoso"]

# Load the reference file
male_heads = pd.read_csv(os.path.join(folder, "muc123a_male_heads.csv"), dtype=str)

# Get all CSV files in the folder except the reference file
csv_files = glob.glob(os.path.join(folder, "*.csv"))
csv_files = [f for f in csv_files if not f.endswith("muc123a_male_heads.csv")]
'''# Uncomment to filter files based on
for file in csv_files:
    if file.endswith('.csv'):
        df = pd.read_csv(file, dtype=str)
        # Check if all key columns exist in the file
        if all(col in df.columns for col in key_cols):
            # Keep only rows with matching keys
            filtered = df.merge(male_heads[key_cols].drop_duplicates(), on=key_cols, how="inner")
            # Save the filtered file (optional: add a suffix or save to another folder)
            filtered.to_csv(file.replace(".csv", "_male_head_filtered.csv"), index=False)
        else:
            print(f"Skipping {file}: missing one or more key columns.")

'''


# Load filtered data for income, expenditure, and wealth
income_df = pd.read_csv(os.path.join(folder, "muc4a_male_head_filtered.csv"), dtype=str)
expenditure_df = pd.read_csv(os.path.join(folder, "hhexpe08_male_head_filtered.csv"), dtype=str)
wealth_a_df = pd.read_csv(os.path.join(folder, "muc6a_male_head_filtered.csv"), dtype=str)
wealth_b_df = pd.read_csv(os.path.join(folder, "muc6b_male_head_filtered.csv"), dtype=str)

# Convert relevant columns to numeric for calculations
for col in ["m4ac11", "m4ac12f", "m4ac22f"]:
    income_df[col] = pd.to_numeric(income_df[col], errors='coerce').fillna(0)
expenditure_df["hhex1nom"] = pd.to_numeric(expenditure_df["hhex1nom"], errors='coerce').fillna(0)
wealth_a_df["m6ac6"] = pd.to_numeric(wealth_a_df["m6ac6"], errors='coerce').fillna(0)
wealth_b_df["m6bc6"] = pd.to_numeric(wealth_b_df["m6bc6"], errors='coerce').fillna(0)

# Aggregate income by key columns
income_agg = income_df.groupby(key_cols)[["m4ac11", "m4ac12f", "m4ac22f"]].sum().reset_index()
income_agg["income"] = income_agg["m4ac11"] + income_agg["m4ac12f"] + income_agg["m4ac22f"]
income_agg = income_agg[key_cols + ["income"]]

# Aggregate expenditure by key columns
expenditure_agg = expenditure_df.groupby(key_cols)["hhex1nom"].sum().reset_index().rename(columns={"hhex1nom": "expenditure"})

# Aggregate wealth by key columns
wealth_a_agg = wealth_a_df.groupby(key_cols)["m6ac6"].sum().reset_index()
wealth_b_agg = wealth_b_df.groupby(key_cols)["m6bc6"].sum().reset_index()
wealth = pd.merge(wealth_a_agg, wealth_b_agg, on=key_cols, how="outer").fillna(0)
wealth["wealth"] = wealth["m6ac6"] + wealth["m6bc6"]
wealth_agg = wealth[key_cols + ["wealth"]]

# Merge all together
result = male_heads[key_cols].drop_duplicates()
result = result.merge(income_agg, on=key_cols, how="left")
result = result.merge(expenditure_agg, on=key_cols, how="left")
result = result.merge(wealth_agg, on=key_cols, how="left")

# Add log income column
result["log_income"] = np.log(result["income"].replace(0, np.nan))

# Add age column from muc123a_male_heads.csv
age_df = pd.read_csv(os.path.join(folder, "muc123a_male_heads.csv"), dtype=str)
age_df = age_df[key_cols + ["m1ac5"]].rename(columns={"m1ac5": "age"})
result = result.merge(age_df, on=key_cols, how="left")

# Save each male head household as a separate CSV file
output_folder = os.path.join(folder, "male_head_households")
os.makedirs(output_folder, exist_ok=True)
for _, row in result.iterrows():
    filename = f"household_{'_'.join([str(row[k]) for k in key_cols])}.csv"
    row_df = pd.DataFrame([row])
    row_df.to_csv(os.path.join(output_folder, filename), index=False)

print(f"Saved {len(result)} files to {output_folder}")

# Combine all household files into one file
household_files = glob.glob(os.path.join(output_folder, "household_*.csv"))
all_households = pd.concat([pd.read_csv(f) for f in household_files], ignore_index=True)
all_households.to_csv(os.path.join(folder, "all_male_head_households.csv"), index=False)
print(f"Combined all households into {os.path.join(folder, 'all_male_head_households.csv')}")