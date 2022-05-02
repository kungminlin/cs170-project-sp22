"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
import math
from pathlib import Path
from typing import Callable, Dict

from instance import Instance
from point import Point
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper

#import numpy as np

def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def penalty(instance, towers):
    """Computes the penalty for this solution."""
    penalty = 0
    for fidx, first in enumerate(towers):
        num_overlaps = 0
        for sidx, second in enumerate(towers):
            if fidx == sidx:
                continue
            if Point.distance_obj(first, second) <= instance.penalty_radius:
                num_overlaps += 1
        penalty += 170 * math.exp(0.17 * num_overlaps)
    return penalty

def penalty_not_sol(instance, towers, new_tower):
    overlap = 0
    for tower in towers:
        if Point.distance_obj(tower, new_tower) <= instance.penalty_radius:
            overlap += 1
    return overlap


def solve_greedy(instance: Instance) -> Solution:
    m = instance.grid_side_length
    n = instance.grid_side_length
    cities = instance.cities
    cities_len = len(instance.cities)
    towers = []
    print(len(cities))
    while(cities):
        cities_to_remove = []
        tower_to_add = Point(x=0,y=0)
        for i in range(m):
            for j in range(n):
                potential_cities = []
                coord = Point(x = i, y = j)
                for city in cities:
                    dist = city.distance_obj(coord)
                    if dist <= instance.coverage_radius:
                        potential_cities.append(city)
                #towers_with_coord = towers + [coord]
                #potential_tower = towers + [tower_to_add]
                if len(potential_cities) > len(cities_to_remove):
                    cities_to_remove = potential_cities
                    tower_to_add = coord
                #if len(cities) <= cities_len*3/5:
                #    if len(potential_cities) >= len(cities_to_remove) and penalty_not_sol(instance, towers, coord) < penalty_not_sol(instance, towers, tower_to_add):
                #        cities_to_remove = potential_cities
                #        tower_to_add = coord
                elif len(potential_cities) >= len(cities_to_remove): 
                    potential_penalty = penalty_not_sol(instance, towers, coord)
                    curr_penalty = penalty_not_sol(instance, towers, tower_to_add)
                    if potential_penalty < curr_penalty:
                        cities_to_remove = potential_cities
                        tower_to_add = coord
                    elif (potential_penalty == curr_penalty) and (coord.x == 0 or coord.y == 0 or coord.x == n-1 or coord.y == n-1 or coord.x == 1 or coord.y == 1 or coord.x == n-2 or coord.y == n-2):
                        cities_to_remove = potential_cities
                        tower_to_add = coord

        for city in cities_to_remove:
            cities.remove(city)
        towers.append(tower_to_add)
    print("\n")
    print(len(towers))
    print(penalty(instance, towers))
    return Solution(
        instance= instance,
        towers = towers,
    )

def solve_greedy_2(instance: Instance) -> Solution:
    # at each step, picks location that covers the most amount of towers with 0 penalty. If 0 penalty is not possible, increase min penalty
    print("solve_greedy_2")
    m = instance.grid_side_length
    n = instance.grid_side_length
    
    cities = instance.cities
    towers = []
    print(len(cities))

    max_penalty = 0
    
    while(cities):
        
        penalty_matrix = np.zeros((n,m))
        covered_cities_matrix = np.zeros((n,m), dtype=object) 

        cities_to_remove = []
        tower_to_add = []

        # for each location of possible tower, 
        for i in range(m):
            for j in range(n):
                coord = Point(x=i, y=j) # tower
                covered_cities = []
                # keep track of number of cities in tower's service radius 
                for city in cities:
                    dist = city.distance_obj(coord) # dist(city, tower)
                    if dist < instance.coverage_radius:
                        covered_cities.append(city)

                covered_cities_matrix[i, j] = covered_cities
                # keep track of penalty 
                towers_with_coord = towers + [coord]
                penalty_matrix[i, j] = penalty(instance, towers_with_coord)

        # look at cities only with min penalty
        max_penalty = np.min(penalty_matrix) # update max_penalty if needed
        print(np.min(penalty_matrix))
        towers_min_penalty = np.transpose(np.where(penalty_matrix <= max_penalty)) # returns all indices where penalty_matrix <= max_penalty
        print(towers_min_penalty)
        # find tower with max city coverage (given min penalty)
        for i in towers_min_penalty: # i is coords of towers with min penalty
            x = i[0]
            y = i[1]
            if (len(covered_cities_matrix[x , y]) >= len(cities_to_remove)):
                if (len(covered_cities_matrix[x, y]) == len(cities_to_remove)): # for some randomization so we don't always pick the first or last location with max city coverage
                    if np.random.random() > 0.5:
                        cities_to_remove = covered_cities_matrix[x, y]
                        tower_to_add = Point(x=x, y=y)
                else:
                    cities_to_remove = covered_cities_matrix[x, y]
                    tower_to_add = Point(x=x, y=y)

        towers.append(tower_to_add)
        for city in cities_to_remove:
            cities.remove(city)
        
        
    print("\n")
    print(len(towers))
    print(penalty(instance, towers))
    return Solution(
        instance= instance,
        towers = towers,
    )  

SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy,
    "greedy2": solve_greedy_2
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
