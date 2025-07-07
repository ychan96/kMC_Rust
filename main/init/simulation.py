import numpy as np
import time
from kmc.utils import identify_final_products

def run_simulation(
    temp_C: float,
    reaction_time: float,
    chain_length: int,
    rate_constants: np.ndarray = None,
    verbose: bool = False
): 
    """
    Run a single KMC simulation.
    
    Parameters:
        temp_C: Temperature in Celsius
        reaction_time: Reaction time in seconds
        chain_length: Initial chain length
        pre_A: Pre-exponential factors
        verbose: Whether to print progress information
    
    Returns:
        dict: Simulation results
    """
    from kmc import KineticMC

    sim = KineticMC(temp_C=temp_C,
                      reaction_time=reaction_time,
                      chain_length=chain_length,
                      rate_constants=rate_constants)
    
    # Only track history if verbose is True and explicitly requested
    history = [] if verbose else None
    steps_performed = 0
    
    start_time = time.time()
    
    # Print initial state if verbose
    if verbose:
        print(f"Starting simulation at {temp_C}°C for {reaction_time}s with chain length {chain_length}")
        print(f"Initial carbon array: {sim.carbon_array}")
        print(f"Initial chain array: {sim.chain_array}")
    
    # Main simulation loop
    while sim.current_time < sim.reaction_time:
        site_counts = sim.update_configuration()
        reaction_idx, dt = sim.select_reaction(site_counts)
        
        if reaction_idx is None:
            if verbose:
                print("No more reactions possible, ending simulation")
            break
        
        sim.current_time += dt
        success = sim.perform_reaction(reaction_idx)
        
        if success:
            steps_performed += 1
            
            # Only log important events or at intervals to reduce overhead
            if verbose and (steps_performed % 1000 == 0 or reaction_idx in [7, 8]):  # Log every 1000 steps or cracking events
                reaction_names = ["Internal adsorption", "Terminal adsorption", 
                                "Internal desorption", "Terminal desorption", 
                                "Internal dehydrogenation", "Terminal dehydrogenation", 
                                "Double M-C desorption", "Internal cracking", "Terminal cracking"]
                
                print(f"Step {steps_performed}, Time: {sim.current_time:.2f}s")
                print(f"Reaction: {reaction_names[reaction_idx]}")
                
                if reaction_idx in [7, 8]:  # Only print arrays for cracking events
                    print(f"Carbon array: {sim.carbon_array}")
                    print(f"Chain array: {sim.chain_array}")
                    
                    # Identify products after cracking
                    products = identify_final_products(sim.carbon_array, sim.chain_array)
                    print(f"Products after cracking: {products}")
                    print("-" * 50)
            
            # Only store history if explicitly requested and verbose
            if history is not None and (steps_performed % 1000 == 0 or reaction_idx in [7, 8]):
                history.append({
                    'time': sim.current_time,
                    'reaction': reaction_idx,
                    'chain': sim.chain_array.copy(),
                    'carbon': sim.carbon_array.copy()
                })
    
    elapsed_time = time.time() - start_time
    products = identify_final_products(sim.carbon_array, sim.chain_array)
    
    if verbose:
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Final time: {sim.current_time:.2f}s, Steps: {steps_performed}")
        print(f"Final products: {products}")
    
    return {
        'carbon_array': sim.carbon_array.copy(),
        'chain_array': sim.chain_array.copy(),
        'time': sim.current_time,
        'history': history,
        'products': products, #list of chain lengths
        'steps': steps_performed,
        'computation_time': elapsed_time
    }


def run_multiple_simulations(
    num_sims: int,
    temp_C: float,
    reaction_time: float,
    chain_length: int,
    rate_constants: np.ndarray = None,
    verbose: bool = False
):
    """
    Run multiple KMC simulations.
    
    Parameters:
        num_sims: Number of simulations to run
        temp_C: Temperature in Celsius
        reaction_time: Reaction time in seconds
        chain_length: Initial chain length
        pre_A: Pre-exponential factors
        verbose: Whether to print progress information
    
    Returns:
        list: List of simulation results
    """
    results = []
    total_start_time = time.time()
    
    print(f"Running {num_sims} simulations at {temp_C}°C...")
    
    for i in range(num_sims):
        if verbose:
            print(f"\nSimulation {i+1}/{num_sims}")
        
        result = run_simulation(
            temp_C=temp_C,
            reaction_time=reaction_time,
            chain_length=chain_length,
            rate_constants=rate_constants,
            verbose=verbose
        )
        
        results.append(result)
        
        # Print brief summary after each simulation
        print(f"Simulation {i+1}: {len(result['products'])} products in {result['steps']} steps ({result['computation_time']:.2f}s)")
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\nAll {num_sims} simulations completed in {total_elapsed_time:.2f} seconds")
    
    return results