"""
This file contains the brute force TSP solver.
Description:
    1. Read data from TSP file
    2. Calculate Euclidean distance between two points
    3. Calculate total distance of a tour
    4. Brute force TSP solver
    5. Write solution to file
"""

import math
import itertools
import time
from tqdm import tqdm


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    :param point1: tuple (x, y)
    :param point2: tuple (x, y)
    :return: euclidean distance
    """
    return round(math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def calculate_total_distance(points, tour):
    """
    Calculate total distance of a tour
    :param points: tuple (x, y)
    :param tour: list of points
    :return: total distance of a tour
    """
    total_distance = 0
    for i in range(len(tour)):
        total_distance += euclidean_distance(points[tour[i]], points[tour[(i + 1) % len(tour)]])
    return total_distance


def brute_force_tsp(points, time_limit):
    """
    Brute force TSP solver
    :param points: tuple (x, y)
    :param time_limit: time limit in seconds
    :return: tuple (shortest tour, shortest distance)
    """
    start_time = time.time()
    shortest_tour = None
    min_distance = float('inf')

    # Might cause overflow!!! (although shows the right progress bar)
    # total_permutations = math.factorial(len(points))
    # progress_bar = tqdm(total=total_permutations, desc="Brute-Force Progress")

    # Use a reasonable threshold for updating the progress bar
    # This won't be the exact number of permutations but will prevent overflow
    update_threshold = 1000
    progress_bar = tqdm(total=update_threshold, desc="Brute-Force Progress")

    for i, tour in enumerate(itertools.permutations(points.keys())):
        # progress_bar.update(1)
        if i % update_threshold == 0:
            progress_bar.update(update_threshold)
            progress_bar.refresh()  # Force an update of the progress bar
        if time.time() - start_time > time_limit:
            print("Time limit exceeded! Program will now exit.")
            break
        total_distance = calculate_total_distance(points, tour)
        if total_distance < min_distance:
            min_distance = total_distance
            shortest_tour = tour

    progress_bar.close()

    return shortest_tour, min_distance
