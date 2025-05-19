"""

run_lcm.py
----------
This code solves the deterministic life cycle model using backward induction.

"""
from numpy import linspace,log, argmax,inf,zeros,seterr, empty,nan,where
from types import SimpleNamespace
import time
from numpy.random import choice,seed
from matplotlib.pyplot import figure,plot,xlabel,ylabel,title,show,legend
from numpy import nanmean
import pandas as pd
import os
import statsmodels.api as sm

seterr(divide='ignore')
seterr(invalid='ignore')

#%% Import from folder
class household():
    '''
    
    Methods:
        __init__(self,**kwargs) -> Set the household's attributes.
        setup(self,**kwargs) -> Sets parameters.
        
    '''
    
    #%% Constructor.
    def __init__(self,**kwargs):
        '''        
        
        This initializes the model.
        
        Optional kwargs:
            All parameters changed by setting kwarg.
            
        '''

        print('--------------------------------------------------------------------------------------------------')
        print('Model')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The model is the deterministic life cycle model and is solved via Backward Induction.')
        
        print('\n--------------------------------------------------------------------------------------------------')
        print('Household')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The household is fintely-lived and dies after T periods.')
        print('   It derives utility from consumption.')
        print('    -> He/she retires after tr periods.')
        
    #%% Set up model.
    def setup(self,**kwargs):
        '''
        
        This sets the parameters and creates the grids for the model.
        
            Input:
                self : Model class.
                kwargs : Values for parameters if not using the default.
                
        '''
        
        # Namespace for parameters, grids, and utility function.
        setattr(self,'par',SimpleNamespace())
        par = self.par

        print('\n--------------------------------------------------------------------------------')
        print('Parameters:')
        print('--------------------------------------------------------------------------------\n')
        
        # Preferences.
        par.beta = 0.96 # Discount factor.
        par.sigma = 2.0 # CRRA.

        # Demographics.
        par.T = 61 # Last period of life.
        par.tr = 41 # First period of retirement.

        # Prices and Income.
        par.r = 0.03 # Interest rate.
        par.ybar = 10.0 # Exogenous income.
        par.kappa = 0.6 # Share of income as pension.

        # Load age-specific income profile from CSV
        try:
            df_income = pd.read_csv(r'c:/Users/Laptop K1/OneDrive/Desktop/Problem_Set_2/VHLSS 2008 Data/avg_log_income_by_age.csv')
            # Expecting columns: 'age', 'avg_log_income' (or similar)
            if 'avg_log_income' in df_income.columns:
                par.Gt = df_income['avg_log_income'].values
            elif 'log_income' in df_income.columns:
                par.Gt = df_income['log_income'].values
            else:
                par.Gt = df_income.iloc[:,1].values  # fallback: use second column
            print('Loaded age-specific income profile from CSV.')
        except Exception as e:
            print('Could not load avg_log_income_by_age.csv:', e)
            par.Gt = None

        # Simulation parameters.
        par.seed_sim = 2025 # Seed for simulation.
        par.TT = 61 # Number of time periods.
        par.NN = 10000 # Number of time periods.

        # Set up asset grid.
        par.alen = 300 # Grid size for a.
        par.amax = 30.0 # Upper bound for a.
        par.amin = 0.0 # Minimum a.
        
        # Update parameter values to kwarg values if you don't want the default values.
        for key,val in kwargs.items():
            setattr(par,key,val)
        
        assert par.beta > 0.0 and par.beta < 1.0
        assert par.sigma >= 1.00
        assert par.ybar > 0.0
        assert par.kappa >= 0.0 and par.kappa <= 1.0
        assert par.alen > 5
        assert par.amax > par.amin
        
        # Set up asset grid.
        par.agrid = linspace(par.amin,par.amax,par.alen) # Equally spaced, linear grid for a (and a').

        # Utility function.
        par.util = util
        
        print('beta: ',par.beta)
        print('sigma: ',par.sigma)
        print('amin: ',par.amin)
        print('amax: ',par.amax)
        print('kappa: ',par.kappa)

#%% CRRA Utility Function.
def util(c,sigma):

    # CRRA utility
    if sigma == 1.0:
        u = log(c) # Log utility.
    else:
        u = (c**(1.0-sigma))/(1.0-sigma) # CRRA utility.
    
    return u

def experience_life(myClass):
    '''
    
    This function solves the deterministic life cycle model.
    
    Input:
        myClass : Model class with parameters, grids, and utility function.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Solving the Model by Backward Induction')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for optimal policy funtions.
    setattr(myClass,'sol',SimpleNamespace())
    sol = myClass.sol

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    
    T = par.T # Last period of life.
    tr = par.tr # First year of retirement.
    
    beta = par.beta # Discount factor.
    sigma = par.sigma # CRRA.
    
    alen = par.alen # Grid size for a.
    agrid = par.agrid # Grid for a (state and choice).

    r = par.r # Real interest rate.
    kappa = par.kappa # Share of income as pension.
    ybar = par.ybar # Exogenous income.

    util = par.util # Utility function.

    # Containers.
    v1 = zeros((alen,T)) # Container for V.
    a1 = zeros((alen,T)) # Container for a'.
    c1 = zeros((alen,T)) # Container for c.

    t0 = time.time()

    for age in reversed(range(T)): # Iterate on the Bellman Equation until convergence.
    
        # Use age-specific income Gt if available, else fallback to ybar
        if hasattr(par, 'Gt') and par.Gt is not None:
            if age < len(par.Gt):
                y_age = par.Gt[age]
            else:
                y_age = par.Gt[-1]
        else:
            y_age = ybar

        if age == T-1: # Python index starts at 0 and does not include the upper limit.
            c1[:,age] = agrid + kappa*y_age
            a1[:,age] = 0.0
            v1[:,age] = util(c1[:,age],sigma)
    
        else:
            if age >= tr: # Retirees get a pension.
                yt = kappa*y_age
            else:
                yt = y_age
    
            for p in range(0,alen): # Loop over the a-states.
                # Consumption.
                ct = agrid[p]+yt-(agrid/(1.0+r))
                ct[ct<0.0] = 0.0
                # Solve the maximization problem.
                vall = util(ct,sigma) + beta*v1[:,age+1] # Compute the value function for each choice of a', given a.
                vall[ct<=0.0] = -inf # Set the value function to negative infinity number when c <= 0.
                v1[p,age] = max(vall) # Maximize: vmax is the maximized value function; ind is where it is in the grid.
                a1[p,age] = agrid[argmax(vall)] # Optimal a'.
                c1[p,age] = ct[argmax(vall)] # Optimal a'.
            
        # Print counter.
        if age%5 == 0:
            print('Age: ',age,'.\n')

    t1 = time.time()
    print('Elapsed time is ',t1-t0,' seconds.')

    # Macro variables, value, and policy functions.
    sol.c = c1 # Consumption policy function.
    sol.a = a1 # Saving policy function.
    sol.v = v1 # Value function.

def grow_old_and_die(myClass):
    '''
    
    This function simulates the deterministic life cycle model.
    
    Input:
        myClass : Model class with parameters, grids, utility function, and policy functions.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Simulate the Model')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for simulation.
    setattr(myClass,'sim',SimpleNamespace())
    sim = myClass.sim

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.

    sigma = par.sigma # CRRA.
    util = par.util # Utility function.
    seed_sim = par.seed_sim # Seed for simulation.

    T = par.T # Last period of life.
    tr = par.tr # First year retired.
    
    ybar = par.ybar # Exogenous income.
    kappa = par.kappa # Share of income as pension.
        
    alen = par.alen # Asset grid size.
    agrid = par.agrid # Assets today (state).
    
    apol = sol.a # Policy function for savings.
    cpol = sol.c # Policy function for consumption.

    TT = par.TT # Time periods.
    NN = par.NN # Population size.
    
    tsim = empty((TT,NN)) * nan # Container for simulated age.
    ysim = empty((TT,NN)) * nan # Container for simulated income.
    csim = empty((TT,NN)) * nan # Container for simulated consumption.
    asim = empty((TT,NN)) * nan # Container for simulated savings.
    usim = empty((TT,NN)) * nan # Container for simulated utility.
            
    # Begin simulation.
    
    seed(seed_sim)

    # All agents start at age 0
    t0_ind = zeros(NN, dtype=int)  # All start at age 0
    a0_ind = choice(linspace(0,alen,alen,endpoint=False,dtype=int),NN) # Index for initial wealth.

    for i in range(0,NN):
        tsim[0,i] = t0_ind[i] # Simulated age.
        # Use age-specific income profile if available
        if hasattr(par, 'Gt') and par.Gt is not None:
            ysim[0,i] = par.Gt[0]
        else:
            ysim[0,i] = ybar
        csim[0,i] = cpol[a0_ind[i],0] # Consumption in period 1 given a0.
        asim[0,i] = apol[a0_ind[i],0] # Savings given a0.
        usim[0,i] = util(csim[0,i],sigma) # Utility in period 1 given a0 and age.

    # Simulate endogenous variables.
    
    for j in range(1,TT): # Time loop.
        for i in range(0,NN): # Person loop.
            if tsim[j-1,i]+1 <= T-1: # Check if still alive.
                tsim[j,i] = tsim[j-1,i]+1 # Age in period t.
                age_idx = int(tsim[j,i])
                # Use age-specific income profile if available
                if hasattr(par, 'Gt') and par.Gt is not None:
                    if age_idx < len(par.Gt):
                        ysim[j,i] = par.Gt[age_idx]
                    else:
                        ysim[j,i] = par.Gt[-1]
                else:
                    if age_idx < tr:
                        ysim[j,i] = ybar
                    else:
                        ysim[j,i] = kappa*ybar
                at_ind = where(asim[j-1,i]==agrid)[0] # Savings choice in the previous period is the state today. Find where the latter is on the grid.
                if at_ind.size > 0:
                    idx = at_ind[0]
                    csim[j,i] = cpol[idx,age_idx] # Consumption in period t.
                    asim[j,i] = apol[idx,age_idx] # Savings for period t+1.
                else:
                    csim[j,i] = nan
                    asim[j,i] = nan
                usim[j,i] = util(csim[j,i],sigma) # Utility in period t.
         
    # Simulated model.
    sim.ysim = ysim # Simulated income.
    sim.tsim = tsim # Simulated age.
    sim.csim = csim # Simulated consumption.
    sim.asim = asim # Simulated savings.
    sim.usim = usim # Simulated utility.

    print('Simulation done.\n')
    print('--------------------------------------------------------------------------------------------------\n')

def track_aging(myClass):
    '''
    
    This function plots the model functions and simulations.
    
    Input:
        myClass : Model class with parameters, grids, utility function, policy functions, and simulations.
        
    '''

    # Model parameters, policy and value functions, and simulations.
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.
    sim = myClass.sim # Simulations.
    
    age = linspace(0,par.T,par.T,endpoint=False,dtype=int)
    
    # Empirical data aggregation
    empirical_folder = r'c:/Users/Laptop K1/OneDrive/Desktop/Problem_Set_2/VHLSS 2008 Data/male_head_households/'
    ages_emp = []
    cons_emp = []
    wealth_emp = []
    
    # Aggregate all CSVs in the folder
    if os.path.exists(empirical_folder):
        for fname in os.listdir(empirical_folder):
            if fname.endswith('.csv'):
                df = pd.read_csv(os.path.join(empirical_folder, fname))
                # Expect columns: 'age', 'consumption', 'wealth' (adjust if different)
                if 'age' in df.columns and 'consumption' in df.columns and 'wealth' in df.columns:
                    ages_emp.extend(df['age'].tolist())
                    cons_emp.extend(df['consumption'].tolist())
                    wealth_emp.extend(df['wealth'].tolist())
    
    # Create DataFrame and compute average by age
    if ages_emp:
        df_emp = pd.DataFrame({'age': ages_emp, 'consumption': cons_emp, 'wealth': wealth_emp})
        emp_c = df_emp.groupby('age')['consumption'].mean()
        emp_a = df_emp.groupby('age')['wealth'].mean()
        emp_ages = emp_c.index.values
    else:
        emp_ages, emp_c, emp_a = [], [], []

    # Plot savings policy function.

    figure(1)
    plot(age,sol.a.T)
    xlabel('$Age$')
    ylabel('$a_{t+1}$') 
    title('Savings Policy Function')

    # Plot consumption policy function.
    
    figure(2)
    plot(age,sol.c.T)
    xlabel('$Age$')
    ylabel('$c_{t}$') 
    title('Consumption Policy Function')
    
    # Plot value function.
    
    figure(3)
    plot(age,sol.v.T)
    xlabel('$Age$')
    ylabel('$V_t(k_t)$') 
    title('Value Function')
    
    # Plot simulated and empirical consumption.
    
    lcp_c = zeros((par.T,1))
    lcp_a = zeros((par.T,1))
    lcp_u = zeros((par.T,1))

    for i in range(par.T):
        lcp_c[i] = nanmean(sim.csim[sim.tsim==i])
        lcp_a[i] = nanmean(sim.asim[sim.tsim==i])
        lcp_u[i] = nanmean(sim.usim[sim.tsim==i])

    figure(4)
    plot(age,lcp_c,label='Model Consumption')
    if len(emp_ages) > 0:
        plot(emp_ages, emp_c, label='Empirical Consumption')
    xlabel('Age')
    ylabel('$c^{sim}_{t}$') 
    title('Simulated vs Empirical Consumption')
    legend()
    
    # Plot simulated and empirical savings/wealth.
    
    figure(5)
    plot(age,lcp_a,label='Model Wealth')
    if len(emp_ages) > 0:
        plot(emp_ages, emp_a, label='Empirical Wealth')
    xlabel('Age')
    ylabel('$a^{sim}_{t+1}$') 
    title('Simulated vs Empirical Wealth')
    legend()

    # Plot simulated utility.
    
    figure(6)
    plot(age,lcp_u)
    xlabel('Age')
    ylabel('$u^{sim}_t$') 
    title('Simulated Utility')
    
    show()

def simulate_and_plot_lifecycle_profiles():
    """
    Simulate the model for different values of beta and gamma, and plot life cycle profiles of consumption and wealth.
    """
    from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, legend, show
    import numpy as np

    age = np.linspace(0, 75, 75, endpoint=False, dtype=int)

    # 1. Vary beta, fix gamma=2.00
    betas = [0.90, 0.92, 0.94, 0.96]
    gamma = 2.00
    fig_c1 = figure()
    fig_a1 = figure()
    for beta in betas:
        model = household()
        model.setup(beta=beta, sigma=gamma, NN=3000, tr=60, T=75)
        experience_life(model)
        grow_old_and_die(model)
        sim = model.sim
        lcp_c = np.zeros((model.par.T, 1))
        lcp_a = np.zeros((model.par.T, 1))
        for i in range(model.par.T):
            lcp_c[i] = np.nanmean(sim.csim[sim.tsim==i])
            lcp_a[i] = np.nanmean(sim.asim[sim.tsim==i])
        fig_c1.gca().plot(age, lcp_c, label=f"β={beta}")
        fig_a1.gca().plot(age, lcp_a, label=f"β={beta}")
    fig_c1.gca().set_xlabel('Age')
    fig_c1.gca().set_ylabel('Average Simulated Consumption')
    fig_c1.gca().set_title('Life Cycle Profile: Simulated Consumption (γ=2.00)')
    fig_c1.gca().legend()
    fig_a1.gca().set_xlabel('Age')
    fig_a1.gca().set_ylabel('Average Simulated Wealth')
    fig_a1.gca().set_title('Life Cycle Profile: Simulated Wealth (γ=2.00)')
    fig_a1.gca().legend()
    show()

    # 2. Vary gamma, fix beta=0.96
    gammas = [2.00, 3.00, 4.00, 5.00]
    beta = 0.96
    fig_c2 = figure()
    fig_a2 = figure()
    for gamma in gammas:
        model = household()
        model.setup(beta=beta, sigma=gamma, NN=3000, tr=60, T=75)
        experience_life(model)
        grow_old_and_die(model)
        sim = model.sim
        lcp_c = np.zeros((model.par.T, 1))
        lcp_a = np.zeros((model.par.T, 1))
        for i in range(model.par.T):
            lcp_c[i] = np.nanmean(sim.csim[sim.tsim==i])
            lcp_a[i] = np.nanmean(sim.asim[sim.tsim==i])
        fig_c2.gca().plot(age, lcp_c, label=f"γ={gamma}")
        fig_a2.gca().plot(age, lcp_a, label=f"γ={gamma}")
    fig_c2.gca().set_xlabel('Age')
    fig_c2.gca().set_ylabel('Average Simulated Consumption')
    fig_c2.gca().set_title('Life Cycle Profile: Simulated Consumption (β=0.96)')
    fig_c2.gca().legend()
    fig_a2.gca().set_xlabel('Age')
    fig_a2.gca().set_ylabel('Average Simulated Wealth')
    fig_a2.gca().set_title('Life Cycle Profile: Simulated Wealth (β=0.96)')
    fig_a2.gca().legend()
    show()

def plot_wealth_heatmap():
    """
    Simulate the model for each combination of gamma and beta, compute average simulated wealth,
    and plot a heatmap with gamma on the x-axis and beta on the y-axis.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    betas = [0.90, 0.92, 0.94, 0.96]
    gammas = [2.00, 3.00, 4.00, 5.00]
    avg_wealth = np.zeros((len(betas), len(gammas)))

    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):
            model = household()
            model.setup(beta=beta, sigma=gamma, NN=3000, tr=60, T=75)
            experience_life(model)
            grow_old_and_die(model)
            sim = model.sim
            # Compute average simulated wealth (ignore NaNs)
            avg_wealth[i, j] = np.nanmean(sim.asim)

    plt.figure(figsize=(7,5))
    im = plt.imshow(avg_wealth, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, label='Average Simulated Wealth')
    plt.xticks(ticks=range(len(gammas)), labels=gammas)
    plt.yticks(ticks=range(len(betas)), labels=betas)
    plt.xlabel('γ (Risk Aversion)')
    plt.ylabel('β (Discount Factor)')
    plt.title('Average Simulated Wealth Heatmap')
    plt.tight_layout()
    plt.show()

#%% Deterministic life cycle Model.
some_dude = household()

# Set the parameters, state space, and utility function.
some_dude.setup(beta = 0.94, sigma = 2.00, NN = 3000) # Set β = 0.94, γ = 2.00, NN = 3000

# Solve the model.
experience_life(some_dude) # Obtain the policy functions for consumption and savings.

# Simulate the model.
grow_old_and_die(some_dude) # Simulate forward in time.

# Graphs.
track_aging(some_dude) # Plot policy functions and simulations.

# Simulate and plot life cycle profiles for different values of beta and gamma.
simulate_and_plot_lifecycle_profiles()

# Plot heatmap of average simulated wealth.
plot_wealth_heatmap()

#%% Empirical analysis: Living area and consumption

def analyze_living_area_consumption():
    """
    Merge household-level data, run regression of consumption on total living area,
    and provide supporting evidence from waterexp and elecexp.
    """
    # File paths
    path_living = r'c:/Users/Laptop K1/OneDrive/Desktop/Problem_Set_2/VHLSS 2008 Data/muc7_male_head_filtered_male_head_filtered.csv'
    path_cons = r'c:/Users/Laptop K1/OneDrive/Desktop/Problem_Set_2/VHLSS 2008 Data/hhexpe08_male_head_filtered_male_head_filtered.csv'

    # Load data
    df_living = pd.read_csv(path_living)
    df_cons = pd.read_csv(path_cons)

    # Merge on household identifiers
    merge_keys = ['tinh', 'huyen', 'xa', 'diaban', 'hoso']
    df_merged = pd.merge(df_cons, df_living[merge_keys + ['m7c2']], on=merge_keys, how='inner')

    # Clean and select variables
    df_merged = df_merged.rename(columns={'m7c2': 'total_living_area'})
    df_merged = df_merged.dropna(subset=['hhex1nom', 'total_living_area'])

    # Prepare regression variables
    y = df_merged['hhex1nom']  # total nominal household consumption
    X = df_merged[['total_living_area']]
    X = sm.add_constant(X)

    # Run regression
    model = sm.OLS(y, X).fit()
    print('\nRegression: Household Consumption on Total Living Area')
    print(model.summary())

    # Supporting evidence: water and electricity expenditure
    if 'waterexp' in df_merged.columns and 'elecexp' in df_merged.columns:
        print('\nCorrelation with water and electricity expenditure:')
        print('Correlation (living area, waterexp):', df_merged['total_living_area'].corr(df_merged['waterexp']))
        print('Correlation (living area, elecexp):', df_merged['total_living_area'].corr(df_merged['elecexp']))
        print('Correlation (consumption, waterexp):', y.corr(df_merged['waterexp']))
        print('Correlation (consumption, elecexp):', y.corr(df_merged['elecexp']))
    else:
        print('waterexp or elecexp not found in merged data.')

    # Brief interpretation
    print('\nInterpretation:')
    print('The coefficient on total living area shows the average increase in household consumption associated with each additional square meter of living space. A positive and significant coefficient suggests that larger living areas are associated with higher household consumption, consistent with the idea that living area proxies for permanent income or wealth. Correlations with water and electricity expenditures provide further evidence that larger homes tend to have higher utility usage, supporting the role of living area as a determinant of household consumption.')

# Run the empirical analysis
after_simulation = True  # Set to True to run after simulation
if after_simulation:
    analyze_living_area_consumption()