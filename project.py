"""
The executable file for the project.
usage:
project.py [-h] -inst filename -alg {BF,Approx,LS} -time cutoff_in_seconds [-seed random_seed]
"""
import argparse
import sys
import os
import time
import traceback

try:
    """
    any non-default import goes here
    """
    import numpy as np
    import random
    from LocalSearchAlg.dataLoader import DataLoader
    from LocalSearchAlg.localSearch import LocalSearch
    from LocalSearch_genetic.LS_genetic_Algorithm import export_solution_file, genetic_algorithm, parse_tsp_file
    from BruteForce.utils import read_data, write_to_file
    from BruteForce.BF import brute_force_tsp
    from approx.approximate_code import main_approx

except Exception:
    traceback.print_exc()


def main(args) -> int:
    """
    Args Parser to parse the command args
    """
    parser = argparse.ArgumentParser(
        description="Different Algorithms to Solve for TSP",
        exit_on_error=False)

    parser.add_argument("-inst", required=True, metavar="filename")
    parser.add_argument("-alg", choices=["BF", "Approx", "LS", "LS_genetic"], required=True)
    parser.add_argument("-time", type=float, required=False, metavar="cutoff_in_seconds")
    parser.add_argument("-seed", type=int, required=False, default=None, metavar="random_seed")
    command_args = parser.parse_args(args)

    if "tsp" in command_args.inst.split("."):
        path = 'DATA/' + command_args.inst
    else:
        path = 'DATA/' + command_args.inst + '.tsp'
    

    if command_args.alg == "LS" or command_args.alg == "LS_genetic":
        if command_args.time == None:
            raise RuntimeError("Please provide a time cutoff for LS or LS_genetic Algorithm.")

    if command_args.alg == "BF":
        time_limit = command_args.time

        points = read_data(path)
        start_time = time.time()

        best_tour, best_distance = brute_force_tsp(points, time_limit)
        elapsed_time = time.time() - start_time
        print("--------------------------------------------------\n",
              "Solution index array: ", best_tour, "\n",
              "Solution cost: ", best_distance, "\n",
              "Time Consumed: ", elapsed_time, "s\n",
              "--------------------------------------------------")
        file_path = command_args.inst
        write_to_file(file_path, time_limit, best_tour, best_distance)

    elif command_args.alg == "Approx":
        # raise NotImplementedError()
        time_limit = command_args.time
        start_time = time.time()
        best_tour, best_distance = main_approx(path, time_limit, command_args.seed)
        end_time = time.time()
        elapsed_time = time.time() - start_time
        print("--------------------------------------------------\n",
              "Solution index array: ", best_tour, "\n",
              "Solution cost: ", best_distance, "\n",
              "Time Consumed: ", elapsed_time, "s\n",
              "--------------------------------------------------")
        # write_to_file(path, time_limit, best_tour, best_distance)

    elif command_args.alg == "LS_genetic":
        if command_args.seed is None:
            command_args.seed = random.randint(1, 10000)

        random.seed(command_args.seed)
        np.random.seed(command_args.seed)

        cities = parse_tsp_file(path)
        start_time = time.time()
        best_path, best_cost = genetic_algorithm(
            cities,
            population_size=100,
            num_generations=1000,
            mutation_rate=0.05,
            seed=command_args.seed,
            cutoff_time=command_args.time
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert best_path to numpy array
        best_path_array = np.array(best_path)
        output_dir = os.path.join(os.path.dirname(__file__))

        export_solution_file(
            best_path,
            best_cost,
            command_args.inst,
            "LS_genetic",
            command_args.time,
            command_args.seed,
            save_dir=output_dir
        )
        print("--------------------------------------------------\n",
              "Solution index array: ", np.array2string(best_path_array), "\n",
              "Solution cost: ", best_cost, "\n",
              "Time Consumed: ", elapsed_time, "s\n",
              "--------------------------------------------------")


    else:
        data = DataLoader(path)
        solver = LocalSearch(data)
        sol, sol_cost, delt_time = solver.solve(seed=command_args.seed,
                                                time_limit=command_args.time)
        file = solver.export_sol_file()
        print("--------------------------------------------------\n",
              "Solution index array: ", np.array2string(sol,separator=',', 
                                                        max_line_width=np.inf), "\n",
              "Solution cost: ", sol_cost, "\n",
              "Time Consumed: ", delt_time, "s\n",
              "--------------------------------------------------")
        print("The solution file is " + file)
    return 0

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        traceback.print_exc()

    os.system("PAUSE")
