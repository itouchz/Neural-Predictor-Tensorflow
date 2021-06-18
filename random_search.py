# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
# implied. See the License for the specific language governing 
# permissions and limitations under the License.

# Modified from google/nasbench repository
# https://github.com/google-research/nasbench

from nasbench import api
import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV1X1, CONV3X3, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

def random_spec(nasbench):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec

def run_random_search(nasbench, max_time_budget=5e6):
    """Run a single roll-out of random search to a fixed time budget."""
    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    while True:
        spec = random_spec(nasbench)
        data = nasbench.query(spec)

        # Perform best search
        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        time_spent, _ = nasbench.get_budget_counters()

        # cumulative time
        times.append(time_spent + times[-1])
        if times[-1] > max_time_budget:
            break

    return times, best_valids, best_tests


def proxylessnas_random(num_reps=1, latency_limit=(-1, -1)):
    choice = np.random.choice

    def help_random():
        arch = [
            {"kernel_size": 3, "expansion": 3},
            {"kernel_size": 5, "expansion": 3},
            {"kernel_size": 7, "expansion": 3},
            {"kernel_size": 3, "expansion": 6},
            {"kernel_size": 5, "expansion": 6},
            {"kernel_size": 7, "expansion": 6},
            {"kernel_size": -1, "expansion": -1}
        ]
        first_layer = list(range(3)) #  [IB3x3, IB5x5, IB7x7] 
        no_zero = list(range(6)) #  [IB3x3-3, IB5x5-3, IB7x7-3, IB3x3-6, IB5x5-6, IB7x7-6]
        with_zero = list(range(7)) # [IB3x3-3, IB5x5-3, IB7x7-3, IB3x3-6, IB5x5-6, IB7x7-6, zero]

        one_hot_rep = (choice(first_layer), 
                       choice(no_zero), choice(with_zero), choice(with_zero), choice(with_zero),
                       choice(no_zero), choice(with_zero), choice(with_zero), choice(with_zero),
                       choice(no_zero), choice(with_zero), choice(with_zero), choice(with_zero),
                       choice(no_zero), choice(with_zero), choice(with_zero), choice(with_zero),
                       choice(no_zero), choice(with_zero), choice(with_zero), choice(with_zero),
                       choice(no_zero))
        return one_hot_rep
    
    rep_set = set()

    while len(rep_set) < num_reps:
        rep = help_random()
        while rep in rep_set:
            rep = help_random()
        rep_set.add(rep)
    return list(rep_set)