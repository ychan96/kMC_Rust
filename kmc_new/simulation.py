import numpy as np
import time
import os
from typing import Optional, Union
import matplotlib.pyplot as plt
from kmc_new.init import BaseKineticMC
from kmc_new.count_sites import ConfigMixin
from kmc_new.reactions import ReactionMixin
from kmc_new.coverage import CoverageMixin
from kmc_new.utils import create_coverage_animation, identify_final_products

class KMC(BaseKineticMC, ConfigMixin, ReactionMixin, CoverageMixin):
    pass

def run_simulation(
    temp_C: float = 250,
    reaction_time: float = 7200,
    m_size: int = 5,
    chain_length: Optional[Union[int, np.ndarray]] = None,
    rate_constants: dict = None,
    verbose: bool = False,
    track_coverage: bool = False,
    max_steps: Optional[int] = None
):
    # Initialize
    sim = KMC(
        temp_C=temp_C,
        reaction_time=reaction_time, 
        chain_length=chain_length, 
        m_size=m_size,
        rate_constants=rate_constants,
    )

    if track_coverage:
        coverage_dir = 'coverage_steps'
        os.makedirs(coverage_dir, exist_ok=True)

    history = [] if verbose else None
    steps_performed = 0
    start_time = time.time()

    # Reaction name mapping for verbose output
    reaction_names = {
        'ads_c1': 'C1 Adsorption',
        'ads_c2': 'C2 Adsorption',
        'ads_c3': 'C3 Adsorption',
        'ads_c4': 'C4 Adsorption',
        'ads_c5plus_internal': 'C5+ Internal Adsorption',
        'ads_c5plus_terminal': 'C5+ Terminal Adsorption',
        'des_c1': 'C1 Desorption',
        'des_c2': 'C2 Desorption',
        'des_c3': 'C3 Desorption',
        'des_c4': 'C4 Desorption',
        'des_c5plus_internal': 'C5+ Internal Desorption',
        'des_c5plus_terminal': 'C5+ Terminal Desorption',
        'dmc_c2_terminal': 'C2 Terminal Double M-C Formation',
        'dmc_c3_terminal': 'C3 Terminal Double M-C Formation',
        'dmc_c4_internal': 'C4 Internal Double M-C Formation',
        'dmc_c4_terminal': 'C4 Terminal Double M-C Formation',
        'dmc_c5plus_internal': 'C5+ Internal Double M-C Formation',
        'dmc_c5plus_terminal': 'C5+ Terminal Double M-C Formation',
        'crk_c2_terminal': 'C2 Terminal Cracking',
        'crk_c3_terminal': 'C3 Terminal Cracking',
        'crk_c4_internal': 'C4 Internal Cracking',
        'crk_c4_terminal': 'C4 Terminal Cracking',
        'crk_c5plus_internal': 'C5+ Internal Cracking',
        'crk_c5plus_terminal': 'C5+ Terminal Cracking',
    }

    # Main loop
    while sim.current_time < sim.reaction_time and (max_steps is None or steps_performed < max_steps):
        # Count available sites
        counts = sim.update_configuration()
        
        # Select reaction
        reaction_key, dt = sim.select_reaction(counts)
        
        if reaction_key is None:
            break
        
        # Perform reaction
        success, chain_info = sim.perform_reaction(reaction_key)
        
        if success:
            # Update surface
            sim.metal_surface(reaction_key, chain_info)
            # Update time
            sim.current_time += dt
            steps_performed += 1

            if track_coverage:
                from kmc_new.utils import plot_surface_coverage
                fig = plot_surface_coverage(sim, save_path=None)
                plt.savefig(f'{coverage_dir}/coverage_step_{steps_performed:04d}.png', dpi=150)
                plt.close(fig)
            
            if verbose:
                products = identify_final_products(sim.chain_array)
                history.append({
                    'step': steps_performed,
                    'time': sim.current_time,
                    'reaction': reaction_names.get(reaction_key, reaction_key), #get(reaction_key, default)
                    'carbon_array': sim.carbon_array.copy(),
                    'chain_array': sim.chain_array.copy(),
                    'products': products
                })
                #if reaction_key.startswith("crk_"):
                print(f"Step {steps_performed}, Time: {sim.current_time:.2f}s")
                print(f"Reaction: {reaction_names.get(reaction_key, reaction_key)}")
                print(f"Carbon array: {sim.carbon_array}")
                print(f"Chain array: {sim.chain_array}")
                print(f"Products after reaction: {products}")
                print("-" * 50)

                # Plot surface coverage
                from kmc_new.utils import plot_surface_coverage
                fig = plot_surface_coverage(sim, save_path=None)
                plt.show(block=False)
                plt.pause(0.1)  # Pause to update the plot

                #Pauses here
                input("Press Enter to continue...")  
                plt.close(fig)

    elapsed_time = time.time() - start_time
    products = identify_final_products(sim.chain_array)

    if verbose:
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Final time: {sim.current_time:.2f}s, Steps: {steps_performed}")
        print(f"Final products: {products}")
        

    return {
        'carbon_array': sim.carbon_array.copy(),
        'chain_array': sim.chain_array.copy(),
        'time': sim.current_time,
        'history': history, 
        'products': products,
        'steps':steps_performed,
        'computation_time':elapsed_time
    }

def run_multiple_simulations(
        num_sims: int,
        temp_C: float,
        reaction_time: float,
        m_size: int,
        chain_length: Optional[Union[int, np.ndarray]] = None,
        rate_constants: np.ndarray = None,
        verbose: bool = False,
        track_coverage: bool = False,
        max_steps: Optional[int] = None
):  

    results = []
    total_start_time = time.time()

    print(f"Running {num_sims} simulations at {temp_C}°C...")

    for i in range(num_sims):
        if verbose:
            print(f"\nSimulation {i+1}/{num_sims}")
        
        #only first simulation
        track = (i == 0) and track_coverage

        result = run_simulation(
            temp_C=temp_C,
            reaction_time=reaction_time,
            m_size=m_size,
            chain_length=chain_length,
            rate_constants=rate_constants,
            verbose=verbose,
            track_coverage=track,
            max_steps=max_steps 
        )

        results.append(result)

        print(f"Simulation {i+1}: {len(result['carbon_array'])} carbon chains"
              f"→ {len(result['products'])} products in {result['steps']} steps"
              f"({result['computation_time']:.2f}s)")
    

    if track_coverage and os.path.exists('coverage_steps'):
        create_coverage_animation('coverage_steps')
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\nAll {num_sims} simulations completed in {total_elapsed_time:.2f} seconds")

    return results
