import numpy as np
import matplotlib.pyplot as plt
import time # To time the simulations for heatmaps

# --- 1. Model Parameters (Original and New) ---
# Beta and Gamma will be varied, so we'll set them inside functions or loops
r = 0.04           # Interest rate
T = 75             # Terminal age
tr = 45            # Retirement age

# Income parameters
rho = 0.9
sigma_epsilon = 0.1
# G_t_profile will be defined later

# Asset grid
num_a_grid = 100 # Consider reducing for faster heatmap generation if needed
a_min = 0.0
a_max = 50.0     # Adjust if C_commit significantly changes wealth levels
a_grid = np.linspace(a_min, a_max, num_a_grid)

# --- NEW PARAMETERS for Committed Housing Expenditure ---
H_living_area = 60.0
# CRITICAL SCALING: Adjust theta_committed_exp based on your G_t income units
# Example: If G_t is in '000s VND and data suggests 75 VND/sqm annual cost
# theta_committed_exp_unscaled = 75.0
# theta_committed_exp = theta_committed_exp_unscaled / 1000.0 # if G_t is in thousands
# If G_t is in actual VND, then theta_committed_exp = 75.0
theta_committed_exp = 75.0 / 10000 # Placeholder: ADJUST THIS CAREFULLY!
C_commit = H_living_area * theta_committed_exp
print(f"MODIFIED MODEL: Annual C_commit: {C_commit:.4f} (in same units as G_t and assets)")


# --- 2. Utility Function ---
def utility(c, gamma_val): # Renamed gamma to gamma_val to avoid conflict
    if c <= 0:
        return -np.inf
    if gamma_val == 1.0:
        return np.log(c)
    else:
        return (c**(1.0 - gamma_val) - 1.0) / (1.0 - gamma_val)

# --- 3. Define G_t_profile (Age-Specific *Gross* Income Component) ---
ages_model = np.arange(T)
G_t_profile = 1.0 + 0.15 * ages_model - 0.0025 * ages_model**2 # Placeholder
G_t_profile[G_t_profile < 0.5] = 0.5
G_t_profile[tr:] = 0
kappa = 0.6

# --- 4. Function to Solve Model (VFI) ---
def solve_model_vfi(beta_val, gamma_val, current_C_commit):
    V = np.zeros((T, num_a_grid))
    C_policy_disc = np.zeros((T, num_a_grid)) # Policy for *discretionary* consumption
    
    income_base_for_pension = G_t_profile[tr-1] if tr > 0 else G_t_profile[0]

    for t_idx in range(T - 1, -1, -1):
        age = ages_model[t_idx]
        for a_idx, a_t in enumerate(a_grid):
            max_val = -np.inf
            best_c_disc = 0

            if age < tr:
                y_t_gross = G_t_profile[t_idx]
            else:
                y_t_gross = kappa * income_base_for_pension
            
            y_t_discretionary = max(0, y_t_gross - current_C_commit)
            cash_on_hand_discretionary = (1 + r) * a_t + y_t_discretionary

            # Search for optimal c_disc over a_next_val choices (more robust)
            for a_next_idx_choice, a_next_val_choice in enumerate(a_grid):
                c_disc_choice = cash_on_hand_discretionary - a_next_val_choice
                
                if c_disc_choice <= 1e-7 : # Consumption must be positive
                    continue

                current_u = utility(c_disc_choice, gamma_val)
                expected_future_v = 0
                if t_idx < T - 1:
                    expected_future_v = np.interp(a_next_val_choice, a_grid, V[t_idx + 1, :])
                
                val = current_u + beta_val * expected_future_v
                if val > max_val:
                    max_val = val
                    best_c_disc = c_disc_choice
            
            V[t_idx, a_idx] = max_val
            C_policy_disc[t_idx, a_idx] = best_c_disc
    return C_policy_disc, V

# --- 5. Function to Simulate Household Paths ---
def simulate_paths(C_policy_disc_input, beta_val, gamma_val, current_C_commit, num_hh=1000): # Reduced num_hh for faster heatmap
    sim_A = np.zeros((num_hh, T + 1))
    sim_C_discretionary = np.zeros((num_hh, T))
    sim_Y_gross = np.zeros((num_hh, T))
    
    np.random.seed(42) # For reproducibility
    epsilon_shocks = np.random.normal(0, sigma_epsilon, (num_hh, T))
    income_base_for_pension_sim = np.zeros(num_hh)


    for h_idx in range(num_hh):
        for t_idx in range(T):
            age = ages_model[t_idx]
            current_assets = sim_A[h_idx, t_idx]

            if age < tr:
                base_gross_income_for_sim = G_t_profile[t_idx]
                y_t_gross_sim = base_gross_income_for_sim * np.exp(epsilon_shocks[h_idx, t_idx] - (sigma_epsilon**2)/2) if base_gross_income_for_sim > 0 else 0
                if age == tr -1 : # Store income for pension calculation
                    income_base_for_pension_sim[h_idx] = y_t_gross_sim

            else:
                y_t_gross_sim = kappa * income_base_for_pension_sim[h_idx]
            
            sim_Y_gross[h_idx, t_idx] = y_t_gross_sim
            y_t_discretionary_sim = max(0, y_t_gross_sim - current_C_commit)
            
            c_discretionary_optimal = np.interp(current_assets, a_grid, C_policy_disc_input[t_idx, :])
            
            cash_on_hand_discretionary_sim = (1 + r) * current_assets + y_t_discretionary_sim
            c_discretionary_optimal = min(c_discretionary_optimal, cash_on_hand_discretionary_sim - a_min) # Ensure a_next >= a_min
            c_discretionary_optimal = max(1e-7, c_discretionary_optimal)

            sim_C_discretionary[h_idx, t_idx] = c_discretionary_optimal
            sim_A[h_idx, t_idx + 1] = cash_on_hand_discretionary_sim - c_discretionary_optimal
            sim_A[h_idx, t_idx + 1] = max(a_min, sim_A[h_idx, t_idx + 1])
            
    sim_C_total = sim_C_discretionary + current_C_commit
    return sim_A[:, :-1], sim_C_discretionary, sim_C_total, sim_Y_gross


# --- 6. Run for a specific Beta and Gamma ---
beta_specific = 0.94
gamma_specific = 2.00

print(f"\n--- Running MODIFIED model for beta={beta_specific}, gamma={gamma_specific} ---")
C_policy_modified, _ = solve_model_vfi(beta_specific, gamma_specific, C_commit)
sim_A_modified, sim_C_disc_modified, sim_C_total_modified, sim_Y_gross_modified = \
    simulate_paths(C_policy_modified, beta_specific, gamma_specific, C_commit, num_hh=3000)

avg_A_modified = np.mean(sim_A_modified, axis=0)
avg_C_disc_modified = np.mean(sim_C_disc_modified, axis=0)
avg_C_total_modified = np.mean(sim_C_total_modified, axis=0)

# Plotting for specific beta and gamma
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(ages_model, avg_A_modified, label=f'Assets (Modified, C_commit={C_commit:.2f})')
plt.xlabel('Age')
plt.ylabel('Average Assets')
plt.title(f'Modified Model: Assets\n(β={beta_specific}, γ={gamma_specific})')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ages_model, avg_C_disc_modified, label=f'Discretionary C (Mod.)')
plt.plot(ages_model, avg_C_total_modified, label=f'Total C (Mod.)', linestyle='--')
plt.axhline(C_commit, color='red', linestyle=':', linewidth=0.8, label=f'C_commit Level')
plt.xlabel('Age')
plt.ylabel('Average Consumption')
plt.title(f'Modified Model: Consumption\n(β={beta_specific}, γ={gamma_specific})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle(f"Life-Cycle Profiles (Modified Model with C_commit = {C_commit:.3f})", fontsize=14)
plt.subplots_adjust(top=0.88)
plt.show()


# --- 7. Generate Heatmap for different Beta and Gamma ---
# Define ranges for beta and gamma as in your original assignment
beta_values = np.array([0.90, 0.92, 0.94, 0.96])
# gamma_values = np.array([2.00, 3.00, 4.00, 5.00]) # Original gamma values from problem set
# If your heatmap was γ = {2.00, 3.00, 4.00,5.00} and β = {0.90,0.92, 0.94, 0.96}
# The problem set image showed y-axis β, x-axis γ. Let's stick to that.
gamma_values_heatmap = np.array([2.00, 3.00, 4.00, 5.00]) # For heatmap x-axis
beta_values_heatmap = np.array([0.90, 0.92, 0.94, 0.96])  # For heatmap y-axis

# Store average wealth for the heatmap
# We want average *simulated* wealth over the individuals' lifetimes (excluding the very start or end if preferred)
# e.g., average wealth between age 20 and tr (retirement)
heatmap_avg_wealth_modified = np.zeros((len(beta_values_heatmap), len(gamma_values_heatmap)))

print("\n--- Generating Heatmap for MODIFIED model ---")
start_heatmap_time = time.time()

for i, beta_h in enumerate(beta_values_heatmap):
    for j, gamma_h in enumerate(gamma_values_heatmap):
        print(f"  Solving for beta={beta_h:.2f}, gamma={gamma_h:.2f}...")
        C_policy_h, _ = solve_model_vfi(beta_h, gamma_h, C_commit)
        sim_A_h, _, _, _ = simulate_paths(C_policy_h, beta_h, gamma_h, C_commit, num_hh=500) # Fewer HH for speed
        
        # Calculate average wealth for this (beta, gamma) pair
        # e.g., average assets over all ages and households
        # Or average of peak wealth, or average wealth during working life
        # Let's take average wealth over the entire simulated life cycle for each household, then average that.
        avg_wealth_for_pair = np.mean(sim_A_h)
        heatmap_avg_wealth_modified[i, j] = avg_wealth_for_pair
        print(f"    Avg Wealth: {avg_wealth_for_pair:.3f}")

end_heatmap_time = time.time()
print(f"Heatmap generation took: {end_heatmap_time - start_heatmap_time:.2f} seconds.")

# Plotting the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_avg_wealth_modified, aspect='auto', origin='lower', cmap='viridis',
           extent=[gamma_values_heatmap[0], gamma_values_heatmap[-1], beta_values_heatmap[0], beta_values_heatmap[-1]])
plt.colorbar(label='Average Simulated Wealth (Modified Model)')
plt.xlabel('γ (Risk Aversion)')
plt.ylabel('β (Discount Factor)')
plt.xticks(gamma_values_heatmap)
plt.yticks(beta_values_heatmap)
plt.title(f'Heatmap of Average Wealth (Modified Model with C_commit={C_commit:.3f})')
plt.show()

# (Assuming solve_model_vfi and simulate_paths are defined as before)

beta_fixed_for_sensitivity = 0.94
gamma_fixed_for_sensitivity = 2.0

# Define different levels of C_commit
# These are illustrative; choose values that make sense for your model's scale
# And reflect plausible variations in H_living_area or theta_committed_exp
C_commit_levels = {
    "Low C_commit (0.25)": 0.250, # Example value
    "Medium C_commit (0.45)": 0.450, # Your current value
    "High C_commit (0.65)": 0.650  # Example value
}

results_sensitivity = {}

print("\n--- Running Sensitivity Analysis for C_commit levels ---")
for label, c_commit_val in C_commit_levels.items():
    print(f"  Solving for {label}...")
    C_policy_sens, _ = solve_model_vfi(beta_fixed_for_sensitivity, gamma_fixed_for_sensitivity, c_commit_val)
    sim_A_sens, sim_C_disc_sens, sim_C_total_sens, _ = \
        simulate_paths(C_policy_sens, beta_fixed_for_sensitivity, gamma_fixed_for_sensitivity, c_commit_val, num_hh=3000)
    
    results_sensitivity[label] = {
        "avg_A": np.mean(sim_A_sens, axis=0),
        "avg_C_disc": np.mean(sim_C_disc_sens, axis=0),
        "avg_C_total": np.mean(sim_C_total_sens, axis=0),
        "c_commit_value": c_commit_val # Store the actual value for plotting
    }

# Now `results_sensitivity` dictionary holds the data for plotting
# Plotting (example for assets, similar for consumption):
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for label, data in results_sensitivity.items():
    plt.plot(ages_model, data["avg_A"], label=f'Assets ({label})')
plt.xlabel('Age')
plt.ylabel('Average Assets')
plt.title(f'Sensitivity to C_commit: Assets\n(β={beta_fixed_for_sensitivity}, γ={gamma_fixed_for_sensitivity})')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for label, data in results_sensitivity.items():
    plt.plot(ages_model, data["avg_C_total"], label=f'Total C ({label})')
    # Optionally plot C_commit lines for each scenario
    plt.axhline(data["c_commit_value"], linestyle=':', linewidth=0.8, label=f'_nolegend_') # No legend for these many lines
plt.xlabel('Age')
plt.ylabel('Average Total Consumption')
plt.title(f'Sensitivity to C_commit: Total Consumption\n(β={beta_fixed_for_sensitivity}, γ={gamma_fixed_for_sensitivity})')
plt.legend() # Might get crowded, consider plotting discretionary C too or separately
plt.grid(True)

plt.tight_layout()
plt.suptitle("Sensitivity of Life-Cycle Profiles to C_commit Level", fontsize=14)
plt.subplots_adjust(top=0.85)
plt.show()
