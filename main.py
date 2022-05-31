import argparse
import json
import multiprocessing
from typing import List

import numpy as np
from tqdm import tqdm

from src.optimize import OptimizationResult, simulated_annealing, SimulatedAnnealingParameters
from src.strategy import GeomStrategy, ExpStrategy, CauchyStrategy, LinearStrategy


def make_strategy(dict_repr):
    if dict_repr['name'] == 'linear':
        return LinearStrategy(dict_repr['beta'])
    elif dict_repr['name'] == 'geom':
        return GeomStrategy(dict_repr['alpha'])
    elif dict_repr['name'] == 'cauchy':
        return CauchyStrategy()
    elif dict_repr['name'] == 'exp':
        return ExpStrategy(dict_repr['c'])

    raise ValueError('Unsupported annealing strategy')


def read_parameters(config_path: str):
    with open(config_path, 'r') as f:
        data = json.load(f)

    data['x_0'] = np.array(data['x_0'])
    data['strategy'] = make_strategy(data['strategy'])

    return SimulatedAnnealingParameters(**data)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', type=str, required=True, help='Path to a config file')
    parser.add_argument('-p', type=int, required=True, help='Number of processes')

    return parser.parse_args()


def print_results(results: List[OptimizationResult]):
    best_sol = min(results, key=lambda res: res.f_val)

    print('Optimization results:')
    print(f'Best solution: f={best_sol.f_val} at x={best_sol.x[0]}, y={best_sol.x[1]}')


def main():
    args = parse_args()
    params = read_parameters(args.c)

    with multiprocessing.Pool(args.p) as pool:
        jobs = pool.imap_unordered(simulated_annealing, [params] * params.n_runs)

        results = [res for res in tqdm(jobs, total=params.n_runs)]

    print_results(results)


if __name__ == '__main__':
    main()
