"""
TSP Approximation Algorithm

This script solves a variation of the Traveling Salesman Problem (TSP) where the edge costs 
are the Euclidean distances between points in a 2D plane.
"""

import sys
import time
from math import sqrt

import networkx as nx
def euclidean_distance(coord1, coord2):
    """Used to determine the edge weights between vertices in the MST construction."""
    return sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def read_dataset(filename):
    """Used to load the vertices from a file, which are then used in constructing the MST and solving the TSP."""
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                vertices.append((float(parts[1]), float(parts[2])))
    return vertices


#Constructs an MST as the basis for the TSP 2-approximation algorithm.

def construct_mst(vertices):
    G = nx.Graph()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            G.add_edge(i + 1, j + 1, weight=euclidean_distance(vertices[i], vertices[j]))


    mst = nx.minimum_spanning_tree(G)
    return mst


#Generates a walk through the MST, which is then used to create a Hamiltonian cycle.

def preorder_walk(mst, start_vertex):
    visited = set()
    walk = []

    def dfs(v):
        visited.add(v)
        walk.append(v)
        for u in mst.neighbors(v):
            if u not in visited:
                dfs(u)

    dfs(start_vertex)
    return walk


#Creates the final TSP tour from the MST pre-order walk

def create_hamiltonian_cycle(preorder_walk):
    visited = set()
    cycle = []

    for vertex in preorder_walk:
        if vertex not in visited:
            visited.add(vertex)
            cycle.append(vertex)

    # Adding the start vertex to complete the cycle

    return cycle



#This is the main function that ties together all steps of the algorithm: it reads the dataset, constructs the MST, performs a pre-order traversal to get a walk, converts the walk into a Hamiltonian cycle, and calculates the total distance of this cycle. It also enforces the time limit for the algorithm.
def main_approx(filename, cut_off_time, random_seed):
    
    vertices = read_dataset(filename)
    start_time = time.time()
    mst = construct_mst(vertices)

    preorder_walk_result = preorder_walk(mst, start_vertex=1)

    route = create_hamiltonian_cycle(preorder_walk_result)

    elapsed_time = time.time() - start_time

    if elapsed_time > cut_off_time:
        print(f"Execution time exceeded the cut-off time of {cut_off_time} seconds.")
        return


    total_distance = sum(
        euclidean_distance(
            vertices[route[i] - 1],  # Adjust index to 0-based
            vertices[route[(i + 1) % len(route)] - 1]  # Adjust index to 0-based
        ) for i in range(len(route))
    )



    # Check if a full tour was computed
    full_tour_computed = len(route) == len(vertices)
    
    output_file = f"{filename.split('.')[0].split('/')[1]}_Approx_{random_seed:.0f}.sol"
    # print(filename.split('.')[0].split('/')[1])
    with open(output_file, 'w') as file:
        file.write(f"{total_distance}\n")
        # Adjust route to 1-based indexing, and avoid repeating the first vertex at the end
        adjusted_route = [r for r in route]
        file.write(','.join(map(str, adjusted_route)))
        #file.write(f"Full Tour Computed: {'Yes' if full_tour_computed else 'No'}\n")


    return route, total_distance
# if __name__ == "__main__":
#     main_approx()
