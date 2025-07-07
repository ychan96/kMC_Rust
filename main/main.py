#!/usr/bin/env python
"""
KMC Simulation for Hydrocarbon Chain Reactions

This script runs Kinetic Monte Carlo simulations to model hydrocarbon chain reactions
and compares the results with experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from datetime import datetime
from kmc import KineticMC, run_simulation, run_multiple_simulations
from kmc.utils import plot_comparison

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run KMC simulations for hydrocarbon chain reactions')
    parser.add_argument('--temp', type=float, default=250, help='Temperature in Celsius (default: 250)')
    parser.add_argument('--time', type=float, default=7200, help='Reaction time in seconds (default: 7200)')
    parser.add_argument('--length', type=int, default=40, help='Initial chain length (default: 40)')
    parser.add_argument('--sims', type=int, default=5, help='Number of simulations to run (default: 5)')
    parser.add_argument('--exp-data', type=str, default='data.xlsx', help='Path to experimental data file')
    parser.add_argument('--verbose', action='store_true', help='Print detailed simulation progress') # not executed unless I call it
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results and plots')
    parser.add_argument('--save-full', action='store_true', help='Save full simulation arrays (larger files)') # not executed unless I call it
    parser.add_argument('--rate-constants', type=str, 
                      help='Comma-separated rate constants (9 values needed)')
    
    args = parser.parse_args()
    
    # Pre-exponential factors (can be adjusted)
    #pre_A = np.array([10, 0.1, 10, 10, 10, 10, 0.00001, 0.003, 0.001])
    
    rate_constants = None
    if args.rate_constants:
        try:
            rate_values = [float(x.strip()) for x in args.rate_constants.split(',')]
            if len(rate_values) != 9:
                print(f"Warning: Expected 9 rate constants, got {len(rate_values)}. Using defaults.")
            else:
                rate_constants = np.array(rate_values)
                print(f"Using provided rate constants: {rate_constants}")
        except ValueError:
            print("Warning: Invalid rate constants format. Using defaults.")
    else:
        # Always initialize with default values if not provided
        rate_constants = np.array([
            1.0e-3,  # Internal adsorption
            1.0e-4,  # Terminal adsorption
            1.0e-5,  # Internal desorption
            1.0e-5,  # Terminal desorption
            1.0e-3,  # Internal dehydrogenation
            1.0e-3,  # Terminal dehydrogenation
            1.0e-8,  # Double M-C desorption
            1.0e-6,  # Internal cracking
            1.0e-7   # Terminal cracking
        ])

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting KMC simulations at {args.temp}°C...")
    print(f"Parameters: time={args.time}s, length={args.length}, sims={args.sims}")
    
    # Run multiple simulations
    results = run_multiple_simulations(
        num_sims=args.sims,
        temp_C=args.temp,
        reaction_time=args.time,
        chain_length=args.length,
        rate_constants=rate_constants,
        verbose=args.verbose
    )
    
    # Extract and save only the essential results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    
    # Collect all products from all simulations
    all_products = []
    for result in results:
        all_products.extend(result['products'])
    
    # Product distribution by carbon number
    from collections import Counter
    product_counts = Counter(all_products)
    
    # Save summary data (much smaller file)
    summary_data = {
        'parameters': {
            'temperature': args.temp,
            'reaction_time': args.time,
            'chain_length': args.length,
            'num_simulations': args.sims,
            'rate_constants' : rate_constants.tolist() if rate_constants is not None else None
            #'pre_exponential_factors': pre_A.tolist()
        },
        'timestamp': timestamp,
        'results_summary': [
            {
                'products': result['products'],
                'steps': result['steps'],
                'time': result['time']
            } for result in results
        ],
        'product_distribution': {str(k): v for k, v in product_counts.items()}
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Only save full arrays if explicitly requested
    if args.save_full:
        full_results_file = os.path.join(args.output_dir, f"full_results_{timestamp}.json")
        with open(full_results_file, 'w') as f:
            serializable_results = []
            for result in results:
                serializable_result = {
                    'carbon_array': result['carbon_array'].tolist(),
                    'chain_array': result['chain_array'].tolist(),
                    'products': result['products'],
                    'time': result['time'],
                    'steps': result['steps']
                }
                serializable_results.append(serializable_result)
            json.dump(serializable_results, f)
        print(f"Full simulation data saved to {full_results_file}")
    
    # Print summary statistics to console
    total_products = len(all_products)
    
    print("\nProduct Distribution:")
    print("Carbon # | Count | Percentage")
    print("-" * 30)
    
    for length in sorted(product_counts.keys()):
        count = product_counts[length]
        percentage = (count / total_products) * 100
        print(f"C{length:<7} | {count:<5} | {percentage:.2f}%")

    max_carbon = max(product_counts.keys()) if product_counts else 0
    
    # Calculate selectivity
    c1_to_c4 = sum(product_counts.get(i, 0) for i in range(1, 5))
    c5_to_c12 = sum(product_counts.get(i, 0) for i in range(5, 13))
    c13_plus = sum(product_counts.get(i, 0) for i in range(13, max_carbon + 1))
    
    print("\nSelectivity:")
    print(f"C1-C4:  {c1_to_c4/total_products*100:.2f}%")
    print(f"C5-C12: {c5_to_c12/total_products*100:.2f}%")
    print(f"C13+:   {c13_plus/total_products*100:.2f}%")
    
    # Create plot comparing to experimental data
    print("\nGenerating comparison plot...")
    try:
        fig = plot_comparison(results, exp_data_file=args.exp_data, chain_length = args.length, simul = args.sims)
        plot_file = os.path.join(args.output_dir, f"comparison_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not create comparison plot: {e}")
    
    print(f"\nSimulation complete. Summary saved to {summary_file}")

if __name__ == "__main__":
    main()