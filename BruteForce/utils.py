"""
This file contains utility functions for BruteForce
Description:
    read_data: read data from TSP file
    write_to_file: write solution to file
"""

def read_data(file_path):
    """
    Read data from TSP file
    :param file_path: path to TSP file
    :return: dictionary of points
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        points = {}
        for line in lines:
            # Skip lines that are not coordinates
            if line.strip().isdigit() or line.startswith("EOF"):
                continue
            parts = line.split()
            if len(parts) == 3 and parts[0].isdigit():
                points[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return points


def write_to_file(path, time_limit, best_tour, best_distance):
    """
    Write solution to file
    :param path: path to TSP file
    :param time_limit: time limit in seconds
    :param best_tour: best tour in list
    :param best_distance: best distance
    :return: none
    """
    # output_file = f"{path.split('.')[0]}_BF_{time_limit}.sol"
    output_file = f"./{path.split('.')[0]}_BF_{time_limit:.0f}.sol"
    with open(output_file, 'w') as file:
        file.write(f"{best_distance}\n")
        file.write(','.join(map(str, best_tour)))
    print("Write successful!")
