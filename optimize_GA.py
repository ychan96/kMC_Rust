#!/usr/bin/env python
"""
Genetic Algorithm Optimization for KMC Rate Constants
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import json
import os
from datetime import datetime
import random

from kmc import run_multiple_simulations

def load_experimental_data(file_path, sheet_name= "Sheet1"):
    try: 
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        valid_data = data[pd.to_numeric(data.iloc[:,0],errors= "coerce" ).notna()]
        
        return valid_data 
    except Exception as e:
        print(f"Error occured: {e}")
        return None



def calculate_fitness(rate_constants, exp_distribution, sim_params):
    """Calculate fitness (1/error) for GA - higher is better"""
    # Run simulations
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
        mass = (14 * length + 2) * count
        mass_by_carbon[length] = mass
    
    total_mass = sum(mass_by_carbon.values())
    if total_mass == 0:
        return 0.001  # Very low fitness for failed simulations
    
    sim_distribution = {int(k): (v / total_mass) * 100 for k, v in mass_by_carbon.items()}
    
    # Calculate RMSE error
    valid_exp_keys = []
    for k in exp_distribution.keys():
        try:
            valid_exp_keys.append(int(k))
        except (ValueError, TypeError):
            continue
            
    carbon_numbers = sorted(set(list(sim_distribution.keys()) + valid_exp_keys))
    
    sim_values = np.array([sim_distribution.get(c, 0) for c in carbon_numbers])
    exp_values = np.array([exp_distribution.get(str(c), 0) for c in carbon_numbers])
    
    rmse = np.sqrt(np.mean((sim_values - exp_values) ** 2))
    
    # Convert to fitness (higher is better)
    fitness = 1.0 / (1.0 + rmse)
    
    print(f"Rate constants: {rate_constants.round(8)}, RMSE: {rmse:.4f}, Fitness: {fitness:.4f}")
    
    return fitness


def selection(population, fitness_scores, num_parents):
    """Tournament selection"""
    parents = []
    for _ in range(num_parents):
        # Tournament size = 3
        tournament_indices = np.random.choice(len(population), 3, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        parents.append(population[winner_idx].copy())
    return np.array(parents)


def crossover(parent1, parent2):
    """Uniform crossover in log space"""
    # Convert to log space
    log_parent1 = np.log10(parent1)
    log_parent2 = np.log10(parent2)
    
    # Uniform crossover
    mask = np.random.rand(9) < 0.5
    child1_log = np.where(mask, log_parent1, log_parent2)
    child2_log = np.where(mask, log_parent2, log_parent1)
    
    # Convert back to linear space
    child1 = np.power(10, child1_log)
    child2 = np.power(10, child2_log)
    
    return child1, child2


def create_population_individual_bounds(pop_size, bounds_list):
    """Create population with individual bounds for each parameter"""
    population = []
    for _ in range(pop_size):
        individual = []
        for bounds in bounds_list:
            # Generate random value in log space for this parameter
            log_value = np.random.uniform(bounds[0], bounds[1])
            # Convert to linear space
            individual.append(np.power(10, log_value))
        population.append(np.array(individual))
    return np.array(population)


def mutate_individual_bounds(individual, bounds_list, mutation_rate=0.1, mutation_strength=0.3):
    """Gaussian mutation with individual bounds"""
    # Convert to log space
    log_individual = np.log10(individual)
    
    # Apply mutation with individual bounds
    for i in range(len(log_individual)):
        if np.random.rand() < mutation_rate:
            log_individual[i] += np.random.normal(0, mutation_strength)
            # Keep within individual bounds
            log_individual[i] = np.clip(log_individual[i], bounds_list[i][0], bounds_list[i][1])
    
    # Convert back to linear space
    return np.power(10, log_individual)

def genetic_algorithm(exp_data_file, pop_size=20, generations=30, sim_params=None):
    """Main GA optimization function"""
    if sim_params is None:
        sim_params = {
            "num_sims": 5,
            "temp_C": 250,
            "reaction_time": 3600,
            "chain_length": 30
        }
    
    # Load experimental data
    exp_data = load_experimental_data(exp_data_file)
    if exp_data is None:
        print("Failed to load experimental data. Exiting.")
        return None
    
    exp_distribution = {str(int(k)): v for k, v in zip(exp_data.iloc[:, 0], exp_data.iloc[:, 1]) 
                       if pd.notnull(k) and str(k) != '--'}
    
    # GA parameters
    bounds_list = [
        [-2, 0],   # k0: Internal adsorption (0.01 to 1.0)
        [-2, 0],   # k1: Terminal adsorption (0.01 to 1.0)
        [-2, 0],   # k2: Internal desorption (0.01 to 1.0)
        [-2, 0],   # k3: Terminal desorption (0.01 to 1.0)
        [-2, 0],   # k4: Internal dehydrogenation (0.01 to 1.0)
        [-2, 0],   # k5: Terminal dehydrogenation (0.01 to 1.0)
        [-5, -3],  # k6: Double M-C desorption (0.00001 to 0.001)
        [-3, -1],  # k7: Internal cracking (0.001 to 0.1)
        [-4, -2]   # k8: Terminal cracking (0.0001 to 0.01)
    ]  
    num_parents = pop_size // 2
    
    # Create initial population
    population = create_population_individual_bounds(pop_size, bounds_list)
    
    # Track best results
    best_fitness_history = []
    best_individual = None
    best_fitness = 0
    
    print("Starting Genetic Algorithm optimization...")
    
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}/{generations}")
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = calculate_fitness(individual, exp_distribution, sim_params)
            fitness_scores.append(fitness)
        
        fitness_scores = np.array(fitness_scores)
        
        # Track best individual
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
        
        best_fitness_history.append(best_fitness)
        
        print(f"Best fitness this generation: {current_best_fitness:.6f}")
        print(f"Overall best fitness: {best_fitness:.6f}")
        
        # Selection
        parents = selection(population, fitness_scores, num_parents)
        
        # Create new population
        new_population = []
        
        # Keep best individuals (elitism)
        elite_size = 2
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            
            # Use the updated mutation function
            child1 = mutate_individual_bounds(child1, bounds_list)
            child2 = mutate_individual_bounds(child2, bounds_list)
            
            new_population.extend([child1, child2])
        
        population = np.array(new_population[:pop_size])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "optimal_rate_constants": best_individual.tolist(),
        "best_fitness": float(best_fitness),
        "best_rmse": float(1.0/best_fitness - 1.0),
        "ga_params": {
            "population_size": pop_size,
            "generations": generations
        },
        "simulation_params": sim_params,
        "fitness_history": best_fitness_history
    }
    
    os.makedirs("ga_results", exist_ok=True)
    results_file = f"ga_results/ga_optimization_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.title('GA Convergence')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.tight_layout()
    plot_file = f"ga_results/ga_convergence_{timestamp}.png"
    plt.savefig(plot_file, dpi=300)
    
    print(f"\nOptimization complete!")
    print(f"Best rate constants: {best_individual}")
    print(f"Best fitness: {best_fitness}")
    print(f"Results saved to {results_file}")
    
    return best_individual

def verify_bounds(individual, bounds_list):
    """Check if individual respects bounds"""
    log_individual = np.log10(individual)
    for i, (low, high) in enumerate(bounds_list):
        if not (low <= log_individual[i] <= high):
            print(f"Parameter k{i} out of bounds: {log_individual[i]} not in [{low}, {high}]")
            return False
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GA optimization for KMC rate constants')
    parser.add_argument('--exp-data', type=str, required=True, help='Path to experimental data file')
    parser.add_argument('--temp', type=float, default=250, help='Temperature in Celsius')
    parser.add_argument('--time', type=float, default=3600, help='Reaction time in seconds')
    parser.add_argument('--length', type=int, default=30, help='Initial chain length')
    parser.add_argument('--sims', type=int, default=5, help='Number of simulations per evaluation')
    parser.add_argument('--pop-size', type=int, default=20, help='Population size')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations')
    
    args = parser.parse_args()
    
    sim_params = {
        "num_sims": args.sims,
        "temp_C": args.temp,
        "reaction_time": args.time,
        "chain_length": args.length
    }
    
    genetic_algorithm(args.exp_data, args.pop_size, args.generations, sim_params)