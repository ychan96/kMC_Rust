# utils.py
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def identify_final_products(carbon_array: np.ndarray, chain_array: np.ndarray):
    products = []
    start_idx = 0
    
    # Iterate through chain_array to find chain breaks 
    for i in range(1,len(chain_array)):
        if chain_array[i] == 0: # Found a break
            # Calculate chain length from start_idx to i
            length = i - start_idx
            if length > 0:
                products.append(length)
            start_idx = i

    if start_idx < len(carbon_array): #last chain
        length = len(carbon_array) - start_idx
        products.append(length)
    
    return products

def plot_comparison(results: List[Dict[str,Any]], exp_data_file='data.xlsx', sheet="Sheet1", max_length=41, chain_length = None, simul=None):
    """
    Plot comparison between simulation results and experimental data.
    
    Parameters:
        results: List of simulation results
        exp_data_file: Excel file containing experimental data
        sheet: Sheet name in the Excel file
        max_length: Maximum chain length to show on x-axis
        chain_length: Initial chain length from the simulation
        simul: Number of simulations to run
    
    Returns:
        matplotlib figure
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    import os

    #two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                  gridspec_kw={'height_ratios': [1, 1]})
    
    if chain_length is None:
        # Fallback value if chain_length cannot be determined
        chain_length = 285  # Default value

    if simul is None:
        # Fallback value if chain_length cannot be determined
        simul = 10  # Default value
    
    # Process simulation results
    all_products = []
    for result in results: #passed as an argument
        products = result['products']
        all_products.extend(products)
    
    # Calculate number-based distribution
    product_counts = Counter(all_products) #{'1':n1, '2':n2, '3':n3, ...}
    total_products = sum(product_counts.values()) # sum([n1,n2,n3..])
    
    all_lengths = list(range(1, max_length + 1))
    num_percentages = [(product_counts.get(length, 0) / total_products * 100) 
                      for length in all_lengths]
    
    # Calculate mass-based distribution
    mass_by_length = {}
    total_mass = 0
    
    for length, count in product_counts.items(): #[('1',n1), ('2',n2), ...]
        # Mass of CnH2n+2 = 14*n + 2 (typical for alkanes)
        product_mass = (14 * length + 2) * count
        mass_by_length[length] = product_mass
        total_mass = (chain_length * 14 + 2)*simul

    mass_percentages = [(mass_by_length.get(length, 0) / total_mass * 100) 
                        if total_mass > 0 else 0 for length in all_lengths]
    
    # Load experimental data
    exp_data = None
    try:
        if os.path.exists(exp_data_file):
            data = pd.read_excel(exp_data_file, header=0, sheet_name=sheet)
            exp_data = data.iloc[0:max_length, 4].values
            
            # Normalize to percentage if necessary
            #if np.sum(exp_data) > 0:
            #   exp_data = (exp_data / np.sum(exp_data)) * 100
    except Exception as e:
        print(f"Warning: Could not load experimental data: {e}")

    # """ Plot number-based distribution (first subplot) """
    width = 0.35  # Width of bars
    
    # Simulation results as bars
    bars1 = ax1.bar([x - width/2 for x in all_lengths], num_percentages, width, 
                   color='steelblue', alpha=0.7, edgecolor='black', 
                   label='Simulation (Number %)')
    
    # Experimental data as line with markers
    #if exp_data is not None and len(exp_data) > 0:
    #    ax1.plot([x + width/2 for x in all_lengths[:len(exp_data)]], exp_data, 
    #            color='red', linestyle='-', linewidth=2, marker='o', 
    #            markersize=5, label='Experimental Data')
        
    # """ Plot mass-based distribution (second subplot) """
    bars2 = ax2.bar([x - width/2 for x in all_lengths], mass_percentages, width,
                   color='green', alpha=0.7, edgecolor='black', 
                   label='Simulation (Mass %)')
    
    # Experimental data again on the second plot if available
    if exp_data is not None and len(exp_data) > 0:
        ax2.plot([x + width/2 for x in all_lengths[:len(exp_data)]], exp_data, 
                color='red', linestyle='-', linewidth=2, marker='o', 
                markersize=5, label='Experimental Data')
        
    # Calculate selectivity metrics for simulation
    if total_products > 0:
        c1_to_c4 = sum(product_counts.get(i, 0) for i in range(1, 5))
        c5_to_c12 = sum(product_counts.get(i, 0) for i in range(5, 13))
        c13_plus = sum(product_counts.get(i, 0) for i in range(13, max_length+1))
        
        # Add text annotations for number-based selectivity
        ax1.text(0.05, 0.87, f"Simulation: C5-C12: {c5_to_c12/total_products*100:.1f}%", #x,y,f-string,   
                transform=ax1.transAxes, va='top', fontsize=11, #relative axes, vertical allignment, fontsize = 11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)) #boxstyle + rounded corner + padding, white fill, gray edge, opacity
        
        # [Additional text annotations for C5-C12 and C13+]
        ax1.text(0.05, 0.95, f"Simulation: C1-C4: {c1_to_c4/total_products*100:.1f}%", 
                transform=ax1.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax1.text(0.05, 0.79, f"Simulation: C13+: {c13_plus/total_products*100:.1f}%", 
                transform=ax1.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Calculate mass-based selectivity metrics
    if total_mass > 0:
        mass_c1_to_c4 = sum(mass_by_length.get(i, 0) for i in range(1, 5))
        mass_c5_to_c12 = sum(mass_by_length.get(i, 0) for i in range(5, 13))
        mass_c13_plus = sum(mass_by_length.get(i, 0) for i in range(13, max_length+1))
        
        # Add text annotations for mass-based selectivity
        ax2.text(0.05, 0.95, f"Simulation: C1-C4: {mass_c1_to_c4/total_mass*100:.1f}%", 
                transform=ax2.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax2.text(0.05, 0.87, f"Simulation: C5-C12: {mass_c5_to_c12/total_mass*100:.1f}%", 
                transform=ax2.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax2.text(0.05, 0.79, f"Simulation: C13+: {mass_c13_plus/total_mass*100:.1f}%", 
                transform=ax2.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
    # Calculate exp_data selectivity and mass-based metrics     
    if exp_data is not None and len(exp_data) >= max_length:
        exp_c1_to_c4 = np.sum(exp_data[:4])
        exp_c5_to_c12 = np.sum(exp_data[4:12])
        exp_c13_plus = np.sum(exp_data[12:])
        
        # text annotations for experimental selectivity (first plot)
        ax1.text(0.65, 0.95, f"Experiment: C1-C4: {exp_c1_to_c4:.1f}%", 
                transform=ax1.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        ax1.text(0.65, 0.87, f"Experiment: C5-C12: {exp_c5_to_c12:.1f}%", 
                transform=ax1.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        ax1.text(0.65, 0.79, f"Experiment: C13+: {exp_c13_plus:.1f}%", 
                transform=ax1.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        # TA for second plot
        ax2.text(0.65, 0.95, f"Experiment: C1-C4: {exp_c1_to_c4:.1f}%", 
                transform=ax2.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        ax2.text(0.65, 0.87, f"Experiment: C5-C12: {exp_c5_to_c12:.1f}%", 
                transform=ax2.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        ax2.text(0.65, 0.79, f"Experiment: C13+: {exp_c13_plus:.1f}%", 
                transform=ax2.transAxes, va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
    # Set titles and labels for number-based plot
    ax1.set_title('Product Distribution by Number: Simulation', fontsize=14)
    ax1.set_xlabel('Carbon Chain Length', fontsize=12)
    ax1.set_ylabel('Yield (%)', fontsize=12)
    ax1.set_xlim(0.5, max_length + 0.5)
    ax1.set_ylim(0, max(max(num_percentages) + 5, 15))
    # exp_data 
    #if exp_data is not None:
    #    ax1.set_ylim(0, max(max(num_percentages) + 5, max(exp_data) + 5, 15))
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.set_xticks(all_lengths[::2])  # Show every other tick to avoid crowding
    ax1.legend(loc='upper right', fontsize=10)
    
    # Set titles and labels for mass-based plot
    ax2.set_title('Product Distribution by Mass: Simulation vs. Experiment', fontsize=14)
    ax2.set_xlabel('Carbon Chain Length', fontsize=12)
    ax2.set_ylabel('Mass Percentage (%)', fontsize=12)
    ax2.set_xlim(0.5, max_length + 0.5)
    ax2.set_ylim(0, max(max(mass_percentages) + 5, 15))
    if exp_data is not None:
        ax2.set_ylim(0, max(max(mass_percentages) + 5, max(exp_data) + 5, 15))
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.set_xticks(all_lengths[::2])  # Show every other tick to avoid crowding
    ax2.legend(loc='upper right', fontsize=10)

     # Add value labels for significant bars (first plot)
    for i, (bar, percentage) in enumerate(zip(bars1, num_percentages)):
        if percentage > 2:  # Only show for bars with significant percentage
            ax1.text(bar.get_x() + bar.get_width()/2, percentage + 0.5, 
                    f"{percentage:.1f}%", ha='center', va='bottom', fontsize=8)
    
    # Add value labels for significant bars (second plot)
    for i, (bar, percentage) in enumerate(zip(bars2, mass_percentages)):
        if percentage > 2:  # Only show for bars with significant percentage
            ax2.text(bar.get_x() + bar.get_width()/2, percentage + 0.5, 
                    f"{percentage:.1f}%", ha='center', va='bottom', fontsize=8)
            
    if exp_data is not None and len(exp_data) > 0:
        # Limit to available experimental data points
        n_points = min(len(exp_data), len(mass_percentages))
        
        # Calculate R^2 for mass distribution
        sim_data = mass_percentages[:n_points]
        exp_data_trim = exp_data[:n_points]
        
        # Filter out zeros to avoid division issues
        valid_indices = [i for i in range(n_points) if exp_data_trim[i] > 0 or sim_data[i] > 0]
        
        if valid_indices:
            exp_valid = np.array([exp_data_trim[i] for i in valid_indices])
            sim_valid = np.array([sim_data[i] for i in valid_indices])
            
            # Calculate R^2
            mean_exp = np.mean(exp_valid)
            ss_tot = np.sum((exp_valid - mean_exp)**2)
            ss_res = np.sum((exp_valid - sim_valid)**2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                ax2.text(0.65, 0.70, f"R² = {r_squared:.3f}", 
                        transform=ax2.transAxes, va='top', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    return fig