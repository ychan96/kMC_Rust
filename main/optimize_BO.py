#!/usr/bin/env python
"""
Bayesian Optimization for KMC Rate Constants

This module optimizes rate constants to match experimental data
using Bayesian optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
import pandas as pd
from collections import Counter
import json
import os
from datetime import datetime

from kmc import run_multiple_simulations
from kmc.utils import identify_final_products


def load_experimental_data(file_path, sheet_name="Sheet1"):
    """Load experimental product distribution data"""
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        # Assuming column structure: carbon_number, distribution_percentage
        return data
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None


def calculate_error(sim_distribution, exp_distribution, metric="rmse"):
    """Calculate error between simulation and experimental distributions"""
    valid_exp_keys = []
    for k in exp_distribution.keys():
        try:
            valid_exp_keys.append(int(k))
        except (ValueError, TypeError):
            # Skip keys that can't be converted to integers
            continue
            
    # Get all valid keys from both distributions
    carbon_numbers = sorted(set(list(sim_distribution.keys()) + valid_exp_keys))
    
    sim_values = np.array([sim_distribution.get(c, 0) for c in carbon_numbers])
    exp_values = np.array([exp_distribution.get(str(c), 0) for c in carbon_numbers])
    
    if metric == "rmse":
        return np.sqrt(np.mean((sim_values - exp_values) ** 2))
    elif metric == "r2":
        corr_matrix = np.corrcoef(sim_values, exp_values)
        return 1 - corr_matrix[0, 1] ** 2  # Return 1-R² so minimizing is better
    else:
        return np.sum(np.abs(sim_values - exp_values))  # MAE


def objective_function(rate_constants_log, exp_distribution, sim_params):
    """Objective function to minimize"""
    # Convert log scale back to linear
    rate_constants = np.power(10, rate_constants_log)
    
    # Run simulations with these rate constants
    results = run_multiple_simulations(
        num_sims=sim_params["num_sims"],
        temp_C=sim_params["temp_C"],
        reaction_time=sim_params["reaction_time"],
        chain_length=sim_params["chain_length"],
        rate_constants=rate_constants,
        verbose=False
    )
    
    # Calculate mass-based distribution
    all_products = []
    for result in results:
        all_products.extend(result['products'])
    
    product_counts = Counter(all_products)
    
    # Convert to mass-based distribution
    mass_by_carbon = {}
    for length, count in product_counts.items():
        mass = (14 * length + 2) * count  # Alkane formula CnH2n+2
        mass_by_carbon[length] = mass
    
    total_mass = sum(mass_by_carbon.values())
    if total_mass > 0:
        mass_distribution = {k: (v / total_mass) * 100 for k, v in mass_by_carbon.items()}
    else:
        mass_distribution = {}
    
    # Calculate error
    error = calculate_error(mass_distribution, exp_distribution, metric="rmse")
    
    # Print current evaluation for monitoring
    print(f"Rate constants: {rate_constants.round(8)}, Error: {error:.4f}")
    
    return error


def optimize_rate_constants(exp_data_file, output_dir="optimization_results", 
                          n_calls=50, sim_params=None):
    """Run Bayesian optimization to find optimal rate constants"""
    if sim_params is None:
        sim_params = {
            "num_sims": 5,
            "temp_C": 250,
            "reaction_time": 3600,
            "chain_length": 30
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experimental data
    exp_data = load_experimental_data(exp_data_file)
    if exp_data is None:
        print("Failed to load experimental data. Exiting.")
        return None
    
    # Convert experimental data to the right format (dict with carbon number as keys)
    exp_distribution = dict(zip(exp_data.iloc[:, 0], exp_data.iloc[:, 1]))
    
    # Define parameter space for the 9 rate constants in log space
    # Typically rate constants span several orders of magnitude
    """                  
    # Internal adsorption
    # Terminal adsorption
    # Internal desorption
    # Terminal desorption
    # Internal dehydrogenation
    # Terminal dehydrogenation
    # Double M-C desorption
    # Internal cracking
    # Terminal cracking
    
    """
    space = [
        Real(-2, 0, name="k0"),     # For 0.07 (log10 ≈ -1.15)  
        Real(-2, 0, name="k1"),     # For 0.07 (log10 ≈ -1.15)  
        Real(-2, 0, name="k2"),     # For 0.06 (log10 ≈ -1.22)  
        Real(-2, 0, name="k3"),     # For 0.06 (log10 ≈ -1.22)
        Real(-2, 0, name="k4"),     # For 0.05 (log10 ≈ -1.30)
        Real(-2, 0, name="k5"),     # For 0.05 (log10 ≈ -1.30)
        Real(-5, -3, name="k6"),    # For 0.00005 (log10 = -4.30)
        Real(-3, -1, name="k7"),    # For 0.0017 (log10 ≈ -2.77)
        Real(-4, -2, name="k8")     # For 0.001 (log10 = -3)
    ]
    
    # Define objective function wrapper
    def objective(params):
        return objective_function(params, exp_distribution, sim_params)
    
    # Run optimization
    print("Starting Bayesian optimization...")
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, 
                        verbose=True, n_jobs=-1)  # Use all available cores
    
    # Extract optimal parameters and convert back to linear scale
    optimal_params = np.power(10, result.x)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
    
    results_data = {
        "optimal_rate_constants": optimal_params.tolist(),
        "optimization_params": {
            "n_calls": n_calls,
            "final_error": float(result.fun)
        },
        "simulation_params": sim_params,
        "all_evaluations": {
            "rate_constants": [np.power(10, x).tolist() for x in result.x_iters],
            "errors": result.func_vals.tolist()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Plot convergence
    fig = plt.figure(figsize=(10, 6))
    plot_convergence(result)
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"convergence_plot_{timestamp}.png")
    plt.savefig(plot_file, dpi=300)
    
    print(f"Optimization complete.")
    print(f"Optimal rate constants: {optimal_params}")
    print(f"Final error: {result.fun}")
    print(f"Results saved to {results_file}")
    print(f"Convergence plot saved to {plot_file}")
    
    return optimal_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize KMC rate constants')
    parser.add_argument('--exp-data', type=str, required=True, help='Path to experimental data file')
    parser.add_argument('--temp', type=float, default=250, help='Temperature in Celsius')
    parser.add_argument('--time', type=float, default=3600, help='Reaction time in seconds')
    parser.add_argument('--length', type=int, default=30, help='Initial chain length')
    parser.add_argument('--sims', type=int, default=5, help='Number of simulations per evaluation')
    parser.add_argument('--n-calls', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--output-dir', type=str, default='optimization_results', 
                       help='Directory to save results and plots')
    
    args = parser.parse_args()
    
    sim_params = {
        "num_sims": args.sims,
        "temp_C": args.temp,
        "reaction_time": args.time,
        "chain_length": args.length
    }
    
    optimize_rate_constants(args.exp_data, args.output_dir, args.n_calls, sim_params)