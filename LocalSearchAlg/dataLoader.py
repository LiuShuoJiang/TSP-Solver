"""
Read the given data file and prepare data for the localserach (Simulated Annealing) algorithm
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances


class DataLoader:
    def __init__(self, path) -> None:
        """
        Read the file and generate vertices and pairwise euclidean distance matrix.
        """
        self.path = path
        self.df = None
        self.dist = None
        
        file = open(path,'r')
        lines = file.readlines()
        file.close()

        df = pd.read_csv(self.path, sep='\t| ',
                         header=None,
                         skiprows=lambda x:lines[x][0].isalpha() and len(lines[x])>1,
                         engine='python')
        df = df.drop(columns=[0])
        df = df.rename(columns={1:'x',2:'y'})
        self.df = df
        points = df.to_numpy()
        self.dist = euclidean_distances(points,points)
        
    def get_points(self) -> np.ndarray:
        """
        Get the vertices in the data file.
        """
        arr = self.df.to_numpy()
        return arr
    
    def get_dist_mat(self) -> np.ndarray:
        """
        Get the pairwise distance matrix.
        """
        return self.dist
    
    def get_dist(self, i:int, j:int) -> float:
        """
        Get the distance given two indices of vetex.
        """
        return self.dist[i][j]
        
    def get_size(self) -> int:
        """
        Get the number of vertices in the data file.
        """
        return self.df.shape[0]
    
    def get_instance(self) -> str:
        """
        Get the name of the file for sol file generation.
        """
        file = os.path.split(self.path)[1]
        filename = file.split(".")[0]
        return filename