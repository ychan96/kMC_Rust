#!/usr/bin/env python
"""
Bayesian Optimization for KMC Rate Constants (16-parameter system)
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

from kmc_new.simulation import run_multiple_simulations


def load_experimental_data(file_path, sheet_name="Sheet1"):
    """Load experimental product distribution data"""
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None


def calculate_rmse(sim_distribution, exp_distribution, max_length=30):
    """Calculate RMSE between simulation and experimental distributions"""
    carbon_numbers = range(1, max_length + 1)
    
    sim_values = np.array([sim_distribution.get(c, 0) for c in carbon_numbers])
    exp_values = np.array([exp_distribution.get(c, 0) for c in carbon_numbers])
    
    return np.sqrt(np.mean((sim_values - exp_values) ** 2))


def objective_function(params, exp_distribution, sim_params):
    """Objective function to minimize"""
    # Convert log-scale params to actual rate constants
    rate_constants = {
        'ads_c1': 10 ** params[0],
        'des_c1': 10 ** params[1],
        'ads_c2': 10 ** params[2],
        'des_c2': 10 ** params[3],
        'ads_c3': 10 ** params[4],
        'des_c3': 10 ** params[5],
        'ads_c4': 10 ** params[6],
        'des_c4': 10 ** params[7],
        'ads_c5plus_internal': 10 ** params[8],
        'des_c5plus_internal': 10 ** params[9],
        'ads_c5plus_terminal': 10 ** params[10],
        'des_c5plus_terminal': 10 ** params[11],
        'dmc_terminal': 10 ** params[12],
        'dmc_internal': 10 ** params[13],
        'crk_terminal': 10 ** params[14],
        'crk_internal': 10 ** params[15],
    }
    
    # Run simulations
    results = run_multiple_simulations(
        num_sims=sim_params["num_sims"],
        temp_C=sim_params["temp_C"],
        reaction_time=sim_params["reaction_time"],
        m_size=sim_params["m_size"],
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
    mass_by_length = {}
    for length, count in product_counts.items():
        if length <= sim_params["max_length"]:
            product_mass = (14 * length + 2) * count
            mass_by_length[length] = product_mass
    
    total_mass = sum(mass_by_length.values())
    if total_mass > 0:
        mass_distribution = {l: (m / total_mass) * 100 for l, m in mass_by_length.items()}
    else:
        mass_distribution = {}
    
    # Calculate RMSE
    error = calculate_rmse(mass_distribution, exp_distribution, sim_params["max_length"])
    
    print(f"Error: {error:.4f}")
    
    return error


def optimize_rate_constants(exp_data_file, 
                            output_dir="optimization_results", 
                            n_calls=50,
                            sim_params=None):  
    """Run Bayesian optimization to find optimal rate constants"""
    if sim_params is None:
        sim_params = {
            "num_sims": 5,
            "temp_C": 250,
            "reaction_time": 7200,
            "chain_length": None,
            "m_size": 5,
            "max_length": 30
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experimental data
    exp_data = load_experimental_data(exp_data_file)
    if exp_data is None:
        print("Failed to load experimental data. Exiting.")
        return None
    
    # Convert experimental data (column 4 has mass percentages)
    exp_distribution = dict(zip(exp_data.iloc[:, 0], exp_data.iloc[:, 4]))
    
    # Define parameter space (log scale for all 16 rate constants)
    space = [
        Real(-4, -2, name="ads_c1"),
        Real(-4, -2, name="des_c1"),
        Real(-4, -2, name="ads_c2"),
        Real(-4, -2, name="des_c2"),
        Real(-6, -2, name="ads_c3"),
        Real(-6, -2, name="des_c3"),
        Real(-4, -2, name="ads_c4"),
        Real(-4, -2, name="des_c4"),
        Real(-4, -2, name="ads_c5plus_internal"),
        Real(-4, -2, name="des_c5plus_internal"),
        Real(-4, -2, name="ads_c5plus_terminal"),
        Real(-4, -2, name="des_c5plus_terminal"),
        Real(-4, -2, name="dmc_terminal"),
        Real(-4, -2, name="dmc_internal"),
        Real(-7, -3, name="crk_terminal"),
        Real(-7, -3, name="crk_internal"),
    ]
    
    def objective(params):
        return objective_function(params, exp_distribution, sim_params)
    
    # Run optimization
    print("Starting Bayesian optimization...")
    result = gp_minimize(
        objective,         
        space, 
        n_calls=n_calls, 
        random_state=42,
        verbose=True, 
        n_jobs=4
    )
    
    # Convert optimal params back to actual values
    optimal_constants = {
        'ads_c1': 10 ** result.x[0],
        'des_c1': 10 ** result.x[1],
        'ads_c2': 10 ** result.x[2],
        'des_c2': 10 ** result.x[3],
        'ads_c3': 10 ** result.x[4],
        'des_c3': 10 ** result.x[5],
        'ads_c4': 10 ** result.x[6],
        'des_c4': 10 ** result.x[7],
        'ads_c5plus_internal': 10 ** result.x[8],
        'des_c5plus_internal': 10 ** result.x[9],
        'ads_c5plus_terminal': 10 ** result.x[10],
        'des_c5plus_terminal': 10 ** result.x[11],
        'dmc_terminal': 10 ** result.x[12],
        'dmc_internal': 10 ** result.x[13],
        'crk_terminal': 10 ** result.x[14],
        'crk_internal': 10 ** result.x[15],
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
    
    results_data = {
        "optimal_rate_constants": optimal_constants,
        "optimization_params": {
            "n_calls": n_calls,
            "final_error": float(result.fun)
        },
        "simulation_params": sim_params,
        "all_evaluations": {
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
    plt.close()
    
    print(f"\nOptimization complete.")
    print(f"Final error: {result.fun:.4f}")
    print(f"\nOptimal rate constants:")
    for key, value in optimal_constants.items():
        print(f"  {key}: {value:.6e}")
    print(f"\nResults saved to {results_file}")
    print(f"Convergence plot saved to {plot_file}")
    
    return optimal_constants


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize KMC rate constants')
    parser.add_argument('--exp-data', type=str, required=True, help='Path to experimental data file')
    parser.add_argument('--temp', type=float, default=250, help='Temperature in Celsius')
    parser.add_argument('--time', type=float, default=7200, help='Reaction time in seconds')
    parser.add_argument('--length', type=int, default=None, help='Initial chain length')
    parser.add_argument('--sims', type=int, default=5, help='Number of simulations per evaluation')
    parser.add_argument('--msize', type=int, default=5, help='Size of metal surface')
    parser.add_argument('--n-calls', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--output-dir', type=str, default='optimization_results', help='Output directory')
    parser.add_argument('--max-length', type=int, default=30, help='Maximum chain length for comparison')
    
    args = parser.parse_args()
    
    sim_params = {
        "num_sims": args.sims,
        "temp_C": args.temp,
        "reaction_time": args.time,
        "chain_length": args.length,
        "m_size": args.msize,
        "max_length": args.max_length
    }
    
    optimize_rate_constants(args.exp_data, args.output_dir, args.n_calls, sim_params)