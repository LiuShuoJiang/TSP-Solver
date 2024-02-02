"""
This is the script to generate the result CSV file (not the .sol file)
for every data using brute-force algorithm.
"""

import os
import time
import csv
from utils import read_data
from BF import brute_force_tsp


def run_brute_force_on_all_datasets(folder_path, time_limit):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".tsp"):
            file_path = os.path.join(folder_path, filename)
            print(f"generate results for {file_path}:")
            points = read_data(file_path)

            start_time = time.time()
            tour, distance = brute_force_tsp(points, time_limit)
            elapsed_time = time.time() - start_time

            full_tour = "Yes" if tour is not None else "No"
            results.append([filename, elapsed_time, distance, full_tour])

    return results


def write_results_to_csv(results, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Time(s)', 'sol.Quality', 'Full_Tour'])
        for row in results:
            writer.writerow(row)


def main():
    folder_path = '../DATA'  # Adjust the path if needed
    time_limit = 300  # Time limit in seconds for each dataset, change it to a small value for testing
    # WARNING: Time consuming!!!
    results = run_brute_force_on_all_datasets(folder_path, time_limit)
    write_results_to_csv(results, 'tsp_brute_force_results.csv')
    print("Write CSV successful!")


if __name__ == "__main__":
    main()
