# TSP-Solver

Required packages for running the `project.py`:

- scikit-learn
- NumPy
- NetworkX
- Pandas
- tqdm

Usage:

```
python project.py [-h] -inst filename -alg {BF,Approx,LS,LS_genetic} [-time cutoff_in_seconds] [-seed random_seed]
```

Example:

```
python project.py -inst Berlin -alg LS -time 4 -seed 0
```

The generated file after running the command would be under the same directory as the `project.py`.
