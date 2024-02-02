"""
File for Simulated Annealing Algorithm to solve TSP problems and generate sol files.
The data for the algorithm is given as the DataLoader object in initialization.
"""
import numpy as np
import numpy.random as random
import time
from LocalSearchAlg.dataLoader import DataLoader 

class LocalSearch:
    def __init__(self, dataloader: DataLoader) -> None:
        self.seed = None
        self.rng = None
        self.data = dataloader
        self.solution = None
        self.cutoff = None
        self.hist = None
       
    def solve(self, seed = None, temp:float=2, decay:float=1e-3, decay_cycle=50, const_k:int = 1e4,
              maxiter:float=1e15, time_limit:float=np.inf, max_stable_iter:float=2e4, history = False) -> tuple:
        """
        Solve the TSP problem and give solution in (solution_array,solution_cost,time_used) tuple.
        """
        self.cutoff = time_limit if time_limit != None else np.inf
        self.seed = seed
        self.rng = random.default_rng(self.seed)
        data_size = self.data.get_size()
        delt_time = 0
        start_time = time.time()
        
        sol = self.__inital_solution()
        sol_cost = self.__cal_total_cost(sol)
        best_sol, best_cost = np.copy(sol),sol_cost
        unchanged_best_iter = 0
        
        run_time = []
        cost_list = []

        for iter in range(int(maxiter)):
            i,j = self.rng.choice(data_size, 2, replace=False)
            cur_cost = self.__update_cost(sol, sol_cost, i, j)
            
            if cur_cost <= sol_cost:
                self.__swap_nodes(sol,i,j)
                sol_cost = cur_cost
                if best_cost > sol_cost:
                    best_sol, best_cost = np.copy(sol),sol_cost
                    unchanged_best_iter = 0
            else:
                deltaE = cur_cost - sol_cost
                p = np.exp(-deltaE / (const_k*temp))
                r = self.rng.random()
                if r < p:
                    sol_cost = cur_cost
                    self.__swap_nodes(sol,i,j)
            
            if iter%decay_cycle == 1: 
                temp = temp * (1-decay)
                # print("current distance is: " , sol_cost)
            
            unchanged_best_iter += 1
            if unchanged_best_iter > max_stable_iter:
                delt_time = time.time()-start_time
                break

            delt_time = time.time()-start_time

            if history == True and iter %1e2 == 1:
                run_time.append(delt_time)
                cost_list.append(best_cost)

            if delt_time >= self.cutoff:
                break
        
        self.hist = (np.array(run_time),np.array(cost_list))
        self.solution = (best_sol,best_cost,delt_time)

        return (best_sol,best_cost,delt_time)

    def __inital_solution(self):
        """
        Give the inital solution to start with.
        """
        idx_arr = np.array(list(range(self.data.get_size())))
        self.rng.shuffle(idx_arr)
        return idx_arr
    

    def __swap_nodes(self, cur_arr:np.array, i:int, j:int) -> None:
        """
        Swap two vertice in the given loop at given indices to generate a new 
        neighbor solution.
        """
    
        cur_arr[i],cur_arr[j] = cur_arr[j],cur_arr[i]
        
        # cur_cost = self.__cal_total_cost(cur_arr)
    
        return
    
    
    def __cal_total_cost(self, arr:np.array) -> float:
        """
        Calculating the total cost of the round trip giving the permutation
        """
        total = [ self.data.get_dist(arr[idx], arr[idx+1]) 
                 for idx in range(0,len(arr)-1) ]
        total = np.sum(np.array(total))
        total = total + self.data.get_dist(arr[-1], arr[0])
        
        return total
    
    def __update_cost(self, arr:np.array, origin_cost:float, i:int, j:int) -> float:
        """
        Updating the cost of the solution after the swap.
        """ 
        tail = len(arr) - 1   
        
        if i > j:
            (i,j)=(j,i)

        if i==0 and j==tail:
            i,j= tail,0
        
        pre_i = i-1 if i!=0 else tail
        pos_i = i+1 if i!=tail else 0

        pre_j = j-1 if j!=0 else tail
        pos_j = j+1 if j!=tail else 0
        
        if pos_i!=j and pre_j!=i:
            
            total = origin_cost - self.data.get_dist(arr[pre_i],arr[i]) - self.data.get_dist(arr[i],arr[pos_i])\
                    - self.data.get_dist(arr[pre_j],arr[j]) - self.data.get_dist(arr[j],arr[pos_j])\
                    + self.data.get_dist(arr[pre_i],arr[j]) + self.data.get_dist(arr[j],arr[pos_i])\
                    + self.data.get_dist(arr[pre_j],arr[i]) + self.data.get_dist(arr[i],arr[pos_j])
        else:
            total = origin_cost - self.data.get_dist(arr[pre_i],arr[i])\
                    - self.data.get_dist(arr[j],arr[pos_j])\
                    + self.data.get_dist(arr[pre_i],arr[j])\
                    +  self.data.get_dist(arr[i],arr[pos_j])
        
        return total
    
    def export_sol_file(self) -> str:
        """
        Export solution as sol file 
        """
        inst = self.data.get_instance()
        if self.cutoff != np.inf:
            file = inst + '_LS_' + "{:.0f}_".format(self.cutoff) +  str(self.seed) + '.sol'
        else:
            file = inst + '_LS_' + "{:.0f}_".format(self.solution[2]) +  str(self.seed) + '.sol'
        sol,sol_cost, _ = self.solution
        
        with open(file, 'w') as my_file:
            my_file.write(str(sol_cost) +'\n' )
            my_file.write(np.array2string(sol+1,separator=',',max_line_width=np.inf).replace(" ","")[1:-1] 
                          + '\n')
        return file
    
    def show_hist(self) -> tuple:
        """
        Show the recorded history in solving the problem.
        """
        return self.hist