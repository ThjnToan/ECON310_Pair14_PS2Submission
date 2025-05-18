import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Load Empirical Data from All Households ---
empirical_folder = 'VHLSS 2008 Data/households_are_here'
if not os.path.exists(empirical_folder):
    raise FileNotFoundError(f"Empirical data folder not found: {empirical_folder}")
empirical_files = [f for f in os.listdir(empirical_folder) if f.endswith('.csv')]

age_col = 'age'
cons_col = 'expenditure'
income_col = 'income'
wealth_col = 'wealth'

empirical_dfs = []
for file in empirical_files:
    df = pd.read_csv(os.path.join(empirical_folder, file))
    # Only keep if all required columns exist
    if all(col in df.columns for col in [age_col, cons_col, income_col, wealth_col]):
        empirical_dfs.append(df[[age_col, cons_col, income_col, wealth_col]])

if not empirical_dfs:
    raise ValueError('No valid household CSVs found with required columns.')

empirical_df = pd.concat(empirical_dfs, ignore_index=True)
empirical_df = empirical_df.dropna(subset=[age_col, cons_col, income_col, wealth_col])

# --- 2. Compute Empirical Statistics by Age ---
empirical_stats = empirical_df.groupby(age_col).agg(
    avg_consumption=(cons_col, 'mean'),
    avg_income=(income_col, 'mean'),
    avg_wealth=(wealth_col, 'mean'),
    var_consumption=(cons_col, 'var')
)
empirical_stats['cons_income_ratio'] = empirical_stats['avg_consumption'] / empirical_stats['avg_income']
empirical_stats['wealth_income_ratio'] = empirical_stats['avg_wealth'] / empirical_stats['avg_income']

# --- 3. Example Simulated Data (Replace with your simulation results) ---
# For demonstration, create dummy arrays. Replace with your actual simulation outputs.
T = 75
ages_model = np.arange(T)
np.random.seed(0)
avg_A_modified = np.random.uniform(1, 10, T)
avg_C_total_modified = np.random.uniform(1, 5, T)
avg_Y_gross_modified = np.random.uniform(2, 8, T)
sim_C_total_modified = np.random.uniform(1, 5, (3000, T))
sim_var_C_total_by_age = np.var(sim_C_total_modified, axis=0)
sim_cons_income_ratio = avg_C_total_modified / avg_Y_gross_modified
sim_wealth_income_ratio = avg_A_modified / avg_Y_gross_modified

# --- 4. Plot Comparison ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Average Consumption by Age: Empirical (left y-axis), Simulated (right y-axis)
ax1 = axes[0, 0]
color_emp = 'tab:blue'
color_sim = 'tab:orange'
ax1.plot(empirical_stats.index, empirical_stats['avg_consumption'], color=color_emp, label='Empirical')
ax1.set_xlabel('Age')
ax1.set_ylabel('Empirical Avg Consumption', color=color_emp)
ax1.tick_params(axis='y', labelcolor=color_emp)
ax1.set_title('Average Consumption by Age')

ax1b = ax1.twinx()
ax1b.plot(ages_model, avg_C_total_modified, color=color_sim, label='Simulated')
ax1b.set_ylabel('Simulated Avg Consumption', color=color_sim)
ax1b.tick_params(axis='y', labelcolor=color_sim)

# Average Wealth by Age: Empirical (left y-axis), Simulated (right y-axis)
ax2 = axes[0, 1]
ax2.plot(empirical_stats.index, empirical_stats['avg_wealth'], color=color_emp, label='Empirical')
ax2.set_xlabel('Age')
ax2.set_ylabel('Empirical Avg Wealth', color=color_emp)
ax2.tick_params(axis='y', labelcolor=color_emp)
ax2.set_title('Average Wealth by Age')

ax2b = ax2.twinx()
ax2b.plot(ages_model, avg_A_modified, color=color_sim, label='Simulated')
ax2b.set_ylabel('Simulated Avg Wealth', color=color_sim)
ax2b.tick_params(axis='y', labelcolor=color_sim)

# Consumption-to-Income Ratio by Age: Empirical (left y-axis), Simulated (right y-axis)
ax3 = axes[1, 0]
ax3.plot(empirical_stats.index, empirical_stats['cons_income_ratio'], color=color_emp, label='Empirical')
ax3.set_xlabel('Age')
ax3.set_ylabel('Empirical C/I Ratio', color=color_emp)
ax3.tick_params(axis='y', labelcolor=color_emp)
ax3.set_title('Consumption-to-Income Ratio by Age')

ax3b = ax3.twinx()
ax3b.plot(ages_model, sim_cons_income_ratio, color=color_sim, label='Simulated')
ax3b.set_ylabel('Simulated C/I Ratio', color=color_sim)
ax3b.tick_params(axis='y', labelcolor=color_sim)

# Wealth-to-Income Ratio by Age: Empirical (left y-axis), Simulated (right y-axis)
ax4 = axes[1, 1]
ax4.plot(empirical_stats.index, empirical_stats['wealth_income_ratio'], color=color_emp, label='Empirical')
ax4.set_xlabel('Age')
ax4.set_ylabel('Empirical W/I Ratio', color=color_emp)
ax4.tick_params(axis='y', labelcolor=color_emp)
ax4.set_title('Wealth-to-Income Ratio by Age')

ax4b = ax4.twinx()
ax4b.plot(ages_model, sim_wealth_income_ratio, color=color_sim, label='Simulated')
ax4b.set_ylabel('Simulated W/I Ratio', color=color_sim)
ax4b.tick_params(axis='y', labelcolor=color_sim)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.suptitle('Empirical vs Simulated: Consumption, Wealth, and Ratios by Age', fontsize=16, y=0.995)
plt.show()

# --- 5. Print Variance of Consumption by Age ---
print("Simulated variance of consumption by age:")
print(sim_var_C_total_by_age)
print("\nEmpirical variance of consumption by age:")
print(empirical_stats['var_consumption'])
