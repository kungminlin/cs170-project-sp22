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


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def penalty(instance, towers):
    """Computes the penalty for this solution."""
    penalty = 0
    print(towers)
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
                towers_with_coord = towers.append(coord)
                potential_tower = towers.append(tower_to_add)
                if (len(potential_cities) > len(cities_to_remove) and penalty(instance, towers_with_coord) < penalty(instance, potential_tower)) or len(cities_to_remove) == 0:
                    cities_to_remove = potential_cities
                    tower_to_add = coord
        for city in cities_to_remove:
            cities.remove(city)
        towers.append(tower_to_add)
    return Solution(
        instance= instance,
        towers = towers,
    )
SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy,
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
