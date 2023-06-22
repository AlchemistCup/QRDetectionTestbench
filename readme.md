# Scrabble Detection Benchmark
This repository contains all the tooling used to test and analyse the performance of various detection approaches for the Scrabble board and rack.

## Installation
After cloning, it's recommended you start by creating a fresh virtual environment with the necessary dependencies. Assuming you're working in a Linux environment, you can accomplish this as follows:
```bash
python3 -m venv .venv # Create venv
source .venv/bin/activate # Activate venv
pip install -r requirements.txt # Install necessary dependencies
```

## Running Benchmarks
There are a variety of benchmarks available to run. In all cases, the number of iterations are configured as a constant in the relevant file. Most benchmarks are relatively short / simple to understand, but this summary below can help you find what you're looking for.
- **Individual QR code detection accuracy:** The [tile directory](tile/) contains multiple benchmarks, each testing the speed of detection of various libraries on a single QR (or micro QR) code. 
- **Board benchmarks:** These testbenches generate random board configurations containing random QR codes, parameterised by % of tiles they contain, which can then be tested on various detection approaches. There are two options for board testing, one for [clean, "perfect" boards](benchmark/board_benchmark.py), and the other testing [a variety of simulated visual artefacts](benchmark/board_reliability_benchmark.py). In both cases, testbenches are run for a particular detection approach by calling the `full_benchmark(n_of_iterations)` from either of these implementations, which tests over a range of tile capacities.
  - The `board_*.py` files provide implementations of the different QR code algorithms considered for a wide range of open source detection libraries
- **Rack benchmarks:** These testbenches generate racks with random QR codes arranged randomly, parameterised by the # of tiles they contain. These are organised similarly to board benchmarks, offering both [perfect](benchmark/rack_benchmark.py) and [non-perfect](benchmark/rack_reliability_benchmark.py) racks to test on.
  - The `racks_*.py` files provide implementations of different different algorithms for the purpose of rack detection.

## Plotting results
The `*_plot.py` files available in the root directory can be used to visulalise data obtained from the benchmarks. These assume your data is stored in the [data directory](data/), and will produce a corresponding graph in the [plots directory](plots/). These directories already contain data obtained during testing.