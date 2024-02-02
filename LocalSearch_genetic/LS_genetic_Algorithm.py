"""
The genetic algorithm is used to optimize the Traveling Salesperson Problem (TSP).
It simulates evolutionary processes to evolve a set of potential solutions over successive generations.
The script includes functions to parse city data from .tsp files, compute paths using genetic operators
(selection, crossover, mutation), and output the best solution to a .sol file. It's designed to
terminate when reaching a user-defined cutoff time or after a certain number of generations.
"""

import numpy as np
import random
import os
import csv
import time
import argparse

# Function to parse command line arguments for random seed
def parse_args():
    parser = argparse.ArgumentParser(description='Genetic Algorithm for TSP')
    parser.add_argument('--seed', type=int, default=None, help='Random seed value (optional)')
    args = parser.parse_args()
    return args.seed

# Function to parse .tsp file and extract the coordinates of the cities
def parse_tsp_file(file_path):
    with open(file_path, 'r') as file:
        # Find the start of the node coordinate section
        while True:
            line = file.readline().strip()
            if line.startswith('NODE_COORD_SECTION'):
                break

        # Read the node coordinates
        cities = []
        while True:
            line = file.readline().strip()
            if line == 'EOF' or not line:
                break
            parts = line.split()
            cities.append((float(parts[1]), float(parts[2])))

    return cities

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate the total length of the path
def path_length(cities, path):
    # Adjust city index for 0-based index array
    return sum([euclidean_distance(cities[path[i]-1], cities[path[i-1]-1]) for i in range(1, len(path))])

# Function to create an initial population of paths
def create_initial_population(num_cities, population_size):
    return [random.sample(range(1, num_cities + 1), num_cities) for _ in range(population_size)]

# Selection function: roulette wheel selection
def select(population, fitness, num_parents):
    fitness_sum = sum(fitness)
    prob = [f / fitness_sum for f in fitness]
    parents_indices = np.random.choice(range(len(population)), size=num_parents, p=prob)
    return [population[i] for i in parents_indices]

# Crossover function: Order Crossover
def crossover(parent1, parent2):
    child = [None] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    child_pos = end
    for gene in parent2:
        if gene not in child:
            if child_pos >= len(parent1):
                child_pos = 0
            child[child_pos] = gene
            child_pos += 1
    return child

# Mutation function: swap two cities
def mutate(path, mutation_rate):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(path) - 1)
            path[i], path[j] = path[j], path[i]
    return path

# Genetic Algorithm implementation
# Genetic Algorithm implementation with cutoff time
def genetic_algorithm(cities, population_size, num_generations, mutation_rate, seed=None, cutoff_time=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = create_initial_population(len(cities), population_size)
    best_path = None
    best_length = float('inf')

    # Start timer for cutoff
    start_time = time.time()

    for generation in range(num_generations):
        # Check for cutoff time
        if cutoff_time is not None and (time.time() - start_time) > cutoff_time:
            print("Cutoff time reached, stopping optimization.")
            break  # Exiting the loop if the cutoff time is reached

        fitness = [1 / path_length(cities, p) for p in population]
        parents = select(population, fitness, population_size // 2)
        next_population = parents.copy()

        # Create the next generation
        for _ in range(len(parents) // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            next_population.append(mutate(child1, mutation_rate))
            next_population.append(mutate(child2, mutation_rate))

        # Select the best path in the next generation
        for path in next_population:
            length = path_length(cities, path)
            if length < best_length:
                best_length = length
                best_path = path

        population = next_population

    return best_path, best_length

# Set a directory for solution files
solution_files_dir = r'F:\Desktop\GT_Study\CSE_6140\Final_Project\github\CSE6140_project\LocalSearch_genetic\Solution_File'
# Create the directory if it does not exist
if not os.path.exists(solution_files_dir):
    os.makedirs(solution_files_dir, exist_ok=True)

# Function to export the solution file
def export_solution_file(best_path, best_cost, tsp_file, method, cutoff, seed=None, save_dir=None):
    if save_dir is None:
        save_dir = solution_files_dir
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # Round the best cost and cutoff to the nearest integer
    best_cost_rounded = int(round(best_cost))
    cutoff_rounded = int(round(cutoff))  # Ensure cutoff is an integer

    # Construct the solution file name
    file_name = f"{tsp_file.split('.')[0]}_{method}_{cutoff_rounded}"
    if seed is not None:
        file_name += f"_{seed}"
    file_name += ".sol"
    # Define the full path for the solution file
    solution_file_path = os.path.join(save_dir, file_name)

    # Write the best solution found to the solution file
    with open(solution_file_path, 'w') as file:
        file.write(f"{best_cost_rounded}\n")  # Line 1: quality of the best solution found
        file.write(','.join(map(str, best_path)))  # Line 2: list of vertex IDs in the TSP tour


# Function to process all .tsp files in a folder and write results to a CSV file
def process_all_tsp_files(folder_path, output_csv, method, cutoff):
    # Ensure the output directory for the CSV file exists
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Prepare the header for the CSV file
    headers = ["Dataset", "Time(s)", "Sol.Quality", "Full Tour", "Etc. Time(s)", "Etc. Sol.Quality", "RelError"]

    # Open the CSV file to write the results
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(headers)

        # Get a list of all .tsp files in the specified folder
        tsp_files = [f for f in os.listdir(folder_path) if f.endswith('.tsp')]

        for tsp_file in tsp_files:
            # Generate a new random seed for each .tsp file
            seed = random.randint(1, 10000)
            random.seed(seed)
            np.random.seed(seed)

            # Construct the full path to the .tsp file
            full_path = os.path.join(folder_path, tsp_file)
            # Parse the .tsp file to get the cities
            cities = parse_tsp_file(full_path)
            # Run the genetic algorithm on the cities and time the execution
            start_time = time.time()
            best_path, best_length = genetic_algorithm(cities, population_size=100, num_generations=1000, mutation_rate=0.05, seed=seed, cutoff_time=cutoff)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Determine if a full tour was computed
            full_tour = "Yes"
            # Write the results for the current .tsp file
            writer.writerow([tsp_file, elapsed_time, best_length, full_tour, "...", "...", "..."])
            # Export the solution file
            export_solution_file(best_path, best_length, tsp_file, method, cutoff, seed)


if __name__ == "__main__":
    # Parse command line arguments for random seed
    seed = parse_args()

    # If no seed is provided, generate a random seed
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)

    # Specify the path to the folder containing the .tsp files
    folder_path = r'F:\Desktop\GT_Study\CSE_6140\Final_Project\DATA\DATA'
    # Specify the path for the output CSV file
    output_csv = r'F:\Desktop\GT_Study\CSE_6140\Final_Project\github\CSE6140_project\LocalSearch_genetic\LS_genetic_results.csv'
    # Process all .tsp files in the folder and write the results to the CSV file
    process_all_tsp_files(folder_path, output_csv, "LS_genetic", 300)

