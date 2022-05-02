"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
import math
from pathlib import Path
from typing import Callable, Dict
import numpy as np
from scipy import optimize

from instance import Instance
from point import Point
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def solve_lp_2(instance: Instance) -> Solution:
    D = instance.grid_side_length
    N = len(instance.cities)
    potential_locations = [[] for city in instance.cities]

    for i in range(D):
        for j in range(D):
            location = Point(x=i, y=j)
            for k in range(N):
                city = instance.cities[k]
                if city.distance_obj(location) <= instance.coverage_radius:
                    potential_locations[k].append(location)
                    break

    flattened_locations = [tower for city in potential_locations for tower in city]
    n = len(flattened_locations)

    penalty_values = np.zeros(n)
    for i in range(n):
        towerA = flattened_locations[i]
        for j in range(n):
            towerB = flattened_locations[j]
            if i != j and towerA.distance_obj(towerB) <= instance.penalty_radius:
                penalty_values[i] += 1

    penalty_values = 170 * np.exp(0.17 * penalty_values)

    X = np.zeros(n)
    c = np.hstack([X, penalty_values])

    A = np.zeros((N+n,2*n))
    b = np.zeros(N+n)

    index = 0
    for k in range(N):
        city_k_locations = potential_locations[k]
        A[k,index:(index + len(city_k_locations))] = -1
        b[k] = -1
        index += len(city_k_locations)

    for i in range(n):
        A[N+i,i] = 1
        A[N+i,n+i] = -1
        b[N+i] = 0

    result = optimize.linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="simplex")

    towers = []
    for i in range(n):
        if np.round(result.x[i]) >= 1:
            towers.append(flattened_locations[i])

    return Solution(
        instance=instance,
        towers=towers
    )


def solve_lp(instance: Instance) -> Solution:
    m = instance.grid_side_length
    n = instance.grid_side_length
    N = len(instance.cities)
    (T, P, C, refresh_row) = setup_constraints(m, n, N)

    # Objective (c.T)
    T_c = np.zeros((m, n))
    P_c = np.ones((m, n, m, n))
    C_c = np.zeros((m, n, N))
    c = flatten(T_c, P_c, C_c)

    # Constraints
    A = []
    b = []

    # City Coverage Constraint
    for i in range(m):
        for j in range(n):
            refresh_row()
            T[i,j] = -1
            C[i,j,:] = 1
            A.append(flatten(T, P, C))
            b.append(0)

    M = np.sqrt(m**2 + n**2)
    for k in range(N):
        for i in range(m):
            for j in range(n):
                refresh_row()
                city = instance.cities[k]
                C[i,j,k] = -M
                A.append(flatten(T, P, C))
                dist = np.sqrt((i - city.x)**2 + (j - city.y)**2)
                b.append(dist - instance.coverage_radius)

                C[i,j,k] = M
                A.append(flatten(T, P, C))
                b.append(M - (dist - instance.coverage_radius))

    for k in range(N):
        refresh_row()
        C[:,:,k] = -1
        A = np.vstack([A, flatten(T, P, C)])
        b.append(-1)

    # Penalty Constraint
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    refresh_row()
                    T[i,j] = 1
                    P[i,j,k,l] = -1
                    A.append(flatten(T, P, C))
                    b.append(0)

                    refresh_row()
                    dist = np.sqrt((i - k)**2 + (j - l)**2)
                    P[i,j,k,l] = -M
                    A.append(flatten(T, P, C))
                    b.append(dist - instance.penalty_radius)

    A = np.array(A)
    b = np.array(b)

    x = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=(0, None))

    x = np.reshape(x, (m, n))

    towers = []
    for i in range(m):
        for j in range(n):
            if np.round(x[i,j]) >= 1:
                towers.append(Point(x=i, y=j))
    
    return Solution(
        instance=instance,
        towers=towers
    )

def flatten(*args):
    return np.hstack([x.flatten() for x in args])

def setup_constraints(m, n, N):
    T = np.zeros((m, n))
    P = np.ones((m, n, m, n))
    C = np.zeros((m, n, N))
    def refresh_row():
        T.fill(0)
        P.fill(0)
        C.fill(0)
    return (T, P, C, refresh_row)

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
def solve_greedy(instance: Instance) -> Solution:
    m = instance.grid_side_length
    n = instance.grid_side_length
    cities = instance.cities
    towers = []
    print(len(cities))
    while(cities):
        cities_to_remove = []
        tower_to_add = Point(x=0,y=0)
        for i in range(m):
            for j in range(n):
                potential_cities = []
                coord = Point(x = i, y=j)
                for city in cities:
                    dist = city.distance_obj(coord)
                    if dist < instance.coverage_radius:
                        potential_cities.append(city)
                towers_with_coord = towers + [coord]
                potential_tower = towers + [tower_to_add]
                if (len(potential_cities) > len(cities_to_remove) and penalty(instance, towers_with_coord) < penalty(instance, potential_tower)) or len(cities_to_remove) == 0:
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
SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy,
    "lp": solve_lp,
    "lp2": solve_lp_2
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
