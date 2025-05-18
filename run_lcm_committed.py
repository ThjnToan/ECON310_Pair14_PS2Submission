import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters (Original and New) ---
beta = 0.96        # Discount factor
gamma = 2.0        # Coefficient of relative risk aversion
r = 0.04           # Interest rate
T = 75             # Terminal age (dies at the beginning of T, lives T periods from 0 to T-1)
tr = 45            # Retirement age (retires at the beginning of tr)

# Income parameters
rho = 0.9          # Persistence of income shocks
sigma_epsilon = 0.1 # Std. dev. of income shocks
# G_t_profile will be defined later (age-specific gross income component)

# Asset grid
num_a_grid = 100
a_min = 0.0        # Borrowing constraint
a_max = 50.0       # Max assets (adjust as needed)
a_grid = np.linspace(a_min, a_max, num_a_grid)

# Consumption grid (for VFI search, if used explicitly)
# num_c_grid = 100
# c_grid = np.linspace(1e-5, a_max, num_c_grid) # Example

# --- NEW PARAMETERS for Committed Housing Expenditure ---
H_living_area = 60.0  # Example: average square meters
theta_committed_exp = 75.0 / 10000 # Example: cost per sqm per year
                             # SCALED to be consistent with income/asset units
                             # E.g., if G_t is in tens of thousands, theta needs to be scaled.
                             # For your regression: 282.91 total. Let's say 75 is committed part.
                             # If income is in '000s VND, and theta is 75 VND/sqm, then theta_scaled = 75/1000.
                             # If income G_t from VHLSS is already in single VND units, then theta=75.
                             # ADJUST THIS SCALING CAREFULLY BASED ON YOUR G_t UNITS!

C_commit = H_living_area * theta_committed_exp # Annual committed housing expenditure

print(f"Annual C_commit: {C_commit:.2f} (in same units as G_t and assets)")

# --- 2. Utility Function ---
def utility(c, gamma):
    if c <= 0:
        return -np.inf
    if gamma == 1.0:
        return np.log(c)
    else:
        return (c**(1.0 - gamma) - 1.0) / (1.0 - gamma)

# --- 3. Define G_t_profile (Age-Specific *Gross* Income Component) ---
# This should be your empirically derived G_t from VHLSS (male heads)
# For example:
# G_t_profile_raw = np.array([... your VHLSS G_t values for ages 0 to T-1 ...])
# Or a functional form if you smoothed it.
# Let's create a placeholder G_t (replace with your actual data)
ages_model = np.arange(T) # Model ages 0 to T-1
# Example: quadratic hump-shape for gross income component, peaking around age 30 in model time
G_t_profile = 1.0 + 0.15 * ages_model - 0.0025 * ages_model**2
G_t_profile[G_t_profile < 0.5] = 0.5 # Ensure minimum income
G_t_profile[tr:] = 0 # No labor income component after retirement age for G_t

# Pension parameter
kappa = 0.6        # Pension replacement rate of income in last working year

# --- 4. Solve Model using Backward Induction (Value Function Iteration) ---
V = np.zeros((T, num_a_grid))
C_policy = np.zeros((T, num_a_grid)) # Policy function for *discretionary* consumption
A_policy = np.zeros((T, num_a_grid)) # Policy function for next period's assets

# Terminal period T-1 (lives for period T-1, decisions made at T-1 for T)
# At T, a_T = 0. So, consume all available resources.
# Gross income in period T-1 (if retired)
# Income at tr-1 to calculate pension:
# This assumes G_t_profile[tr-1] is representative of y_{t_r-1} for pension calculation.
# A more accurate way might be to use an expected income at t_r-1.
# For simplicity, let's assume G_t_profile[tr-1] is the relevant base for pension.
# If tr-1 < 0 (model starts in retirement), this needs adjustment.
income_base_for_pension = G_t_profile[tr-1] if tr > 0 else G_t_profile[0]


for t_idx in range(T - 1, -1, -1):
    age = ages_model[t_idx]
    for a_idx, a_t in enumerate(a_grid):
        max_val = -np.inf
        best_c = 0
        best_a_next = 0

        # Determine gross income for the current period t_idx
        if age < tr: # Working
            y_t_gross = G_t_profile[t_idx] # Using deterministic G_t for VFI simplicity.
                                         # Could add expectation over shocks if model is richer.
        else: # Retired
            y_t_gross = kappa * income_base_for_pension

        # --- MODIFICATION: Calculate discretionary income ---
        y_t_discretionary = max(0, y_t_gross - C_commit) # Ensure non-negative

        # Cash on hand for discretionary spending/saving
        cash_on_hand_discretionary = (1 + r) * a_t + y_t_discretionary

        # Iterate over possible consumption choices (or use an optimizer)
        # This simplified loop assumes consumption is a fraction of CoH or iterates over a c_grid
        # A more robust way is to iterate over next period's assets a_next_val from a_grid
        for c_choice in np.linspace(1e-5, cash_on_hand_discretionary, 50): # Discretionary consumption
            if c_choice <= 0: continue

            a_next_val = cash_on_hand_discretionary - c_choice

            if a_next_val < a_min - 1e-7: # Check against borrowing constraint (allow for small numerical error)
                continue # Infeasible
            # Ensure a_next_val is on the grid for interpolation or find closest
            a_next_idx = np.argmin(np.abs(a_grid - a_next_val))
            # More precise would be to interpolate V[t_idx + 1, :]

            current_u = utility(c_choice, gamma)
            expected_future_v = 0
            if t_idx < T - 1:
                # Interpolate value function for a_next_val
                expected_future_v = np.interp(a_next_val, a_grid, V[t_idx + 1, :])

            val = current_u + beta * expected_future_v

            if val > max_val:
                max_val = val
                best_c = c_choice
                best_a_next = a_next_val
        
        V[t_idx, a_idx] = max_val
        C_policy[t_idx, a_idx] = best_c       # Stores discretionary consumption
        A_policy[t_idx, a_idx] = best_a_next

# --- 5. Simulate Household Paths (Forward Iteration) ---
num_households = 3000
sim_A = np.zeros((num_households, T + 1)) # Assets from age 0 to T
sim_C_discretionary = np.zeros((num_households, T)) # *Discretionary* Consumption from age 0 to T-1
sim_Y_gross = np.zeros((num_households, T))    # Gross income
sim_Y_discretionary = np.zeros((num_households, T))# Discretionary income

# Initial conditions (e.g., zero assets at age 0)
# sim_A[:, 0] = 0.0 # Default is zeros

# For stochastic income simulation part
np.random.seed(42)
epsilon_shocks = np.random.normal(0, sigma_epsilon, (num_households, T))
sim_Log_Y_persistent_component = np.zeros((num_households, T))


for h_idx in range(num_households):
    # Initial income realization (e.g., from G_0 * exp(epsilon))
    # For simplicity, let first period persistent component be based on G_0 with a shock
    if G_t_profile[0] > 0: # Avoid log(0) if G_0 is 0
         sim_Log_Y_persistent_component[h_idx, 0] = np.log(G_t_profile[0]) + epsilon_shocks[h_idx, 0]
    else:
         sim_Log_Y_persistent_component[h_idx, 0] = epsilon_shocks[h_idx, 0] # if G_0 is 0, income is just shock related

    for t_idx in range(T): # Loop from age 0 to T-1
        age = ages_model[t_idx]
        current_assets = sim_A[h_idx, t_idx]

        # Determine gross income
        if age < tr: # Working
            if t_idx > 0:
                 sim_Log_Y_persistent_component[h_idx, t_idx] = rho * sim_Log_Y_persistent_component[h_idx, t_idx-1] + epsilon_shocks[h_idx, t_idx]
            # Gross income combines deterministic G_t and persistent stochastic part
            # This formulation assumes G_t is additive in log, or multiplicative directly.
            # Let's use G_t * exp(persistent_log_income_shock_part)
            # Ensure G_t_profile[t_idx] is the mean/base for this age.
            # The VFI used a simpler y_t_gross = G_t_profile[t_idx]. For consistency,
            # the simulation's income process should align with VFI assumptions, or VFI needs richer income states.
            # For now, let's use a common income simulation:
            base_gross_income_for_sim = G_t_profile[t_idx] # from empirical profile
            if base_gross_income_for_sim > 0: # only apply shock if base > 0
                 y_t_gross_sim = base_gross_income_for_sim * np.exp(epsilon_shocks[h_idx, t_idx] - (sigma_epsilon**2)/2) # Lognormal mean adjustment
            else:
                 y_t_gross_sim = 0
            # A more common approach for simulation:
            # log_y_t = np.log(G_t_profile[t_idx]) + persistent_component_for_age_t
            # y_t_gross_sim = np.exp(log_y_t)

        else: # Retired
            # Pension based on income in the last working year (tr-1)
            # We need the gross income from tr-1 for this household
            income_at_retirement_for_pension = sim_Y_gross[h_idx, tr-1] if tr > 0 else G_t_profile[0] # Fallback if tr=0
            y_t_gross_sim = kappa * income_at_retirement_for_pension
        
        sim_Y_gross[h_idx, t_idx] = y_t_gross_sim

        # --- MODIFICATION: Calculate discretionary income for simulation ---
        y_t_discretionary_sim = max(0, y_t_gross_sim - C_commit)
        sim_Y_discretionary[h_idx, t_idx] = y_t_discretionary_sim

        # Get optimal *discretionary* consumption from policy function
        # Interpolate C_policy based on current_assets and current age t_idx
        # The policy function was solved on a_grid and for each age t_idx
        # It implicitly used G_t_profile[t_idx] (potentially adjusted by C_commit) as the income state.
        # Since our VFI's income state didn't explicitly include the stochastic part, this is an approximation.
        c_discretionary_optimal = np.interp(current_assets, a_grid, C_policy[t_idx, :])
        
        # Ensure consumption is affordable from discretionary resources
        cash_on_hand_discretionary_sim = (1 + r) * current_assets + y_t_discretionary_sim
        c_discretionary_optimal = min(c_discretionary_optimal, cash_on_hand_discretionary_sim)
        c_discretionary_optimal = max(1e-7, c_discretionary_optimal) # Ensure positive

        sim_C_discretionary[h_idx, t_idx] = c_discretionary_optimal

        # Calculate next period's assets
        sim_A[h_idx, t_idx + 1] = cash_on_hand_discretionary_sim - c_discretionary_optimal
        # Ensure assets meet borrowing constraint
        sim_A[h_idx, t_idx + 1] = max(a_min, sim_A[h_idx, t_idx + 1])


# --- 6. Analyze and Plot Results ---
sim_C_total = sim_C_discretionary + C_commit # Calculate total consumption

avg_A = np.mean(sim_A[:, :-1], axis=0) # Average assets over T periods
avg_C_discretionary = np.mean(sim_C_discretionary, axis=0)
avg_C_total = np.mean(sim_C_total, axis=0)
avg_Y_gross = np.mean(sim_Y_gross, axis=0)
avg_Y_discretionary = np.mean(sim_Y_discretionary, axis=0)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(ages_model, avg_A, label='Average Assets (Simulated)')
plt.xlabel('Age')
plt.ylabel('Assets')
plt.title('Simulated Life-Cycle Assets')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(ages_model, avg_C_discretionary, label='Avg. Discretionary C (Sim)')
plt.plot(ages_model, avg_C_total, label='Avg. Total C (Sim)', linestyle='--')
plt.xlabel('Age')
plt.ylabel('Consumption')
plt.title('Simulated Life-Cycle Consumption')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(ages_model, avg_Y_gross, label='Avg. Gross Income (Sim)')
plt.plot(ages_model, avg_Y_discretionary, label='Avg. Discretionary Income (Sim)', linestyle=':')
plt.plot(ages_model, G_t_profile, label='G_t Profile (Input)', linestyle='-.')
plt.axhline(C_commit, color='red', linestyle='--', linewidth=0.8, label=f'C_commit ({C_commit:.2f})')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Simulated Life-Cycle Income')
plt.legend()
plt.grid(True)

# Policy function plot (Discretionary Consumption) for a specific age (e.g., age 20)
age_for_policy_plot = 20
plt.subplot(2, 2, 4)
plt.plot(a_grid, C_policy[age_for_policy_plot, :], label=f'Discretionary C Policy (Age {age_for_policy_plot})')
plt.xlabel('Assets')
plt.ylabel('Discretionary Consumption')
plt.title('Consumption Policy Function')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Reminder: C_commit was {C_commit}. Ensure this value and theta_committed_exp scaling are correct for your income units.")