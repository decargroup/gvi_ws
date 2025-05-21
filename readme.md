# GVI Workspace

A Python package for graph-based variational inference (GVI) in robotics and state estimation. This workspace includes simulation tools, models, and benchmarking utilities built around the [`navlie`](https://github.com/decargroup/navlie) library and other scientific Python tools.

## Project Structure
```
gvi-ws/
├── src/
│   └── gvi/               # Core implementation
├── tests/                 # Unit tests
├── examples/              # Example scripts and experiments
├── figs/                  # Generated plots and figures
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # You're here
```

## Installation

We recommend using a virtual environment:

```bash
# Create and activate environment (optional)
python -m venv .venv
source .venv/bin/activate 

# Install dependencies
pip install -e .
```
## Getting Started

Run an example script:

```bash
python examples/se2_localization.py
```
## Figures and Results

Plots and benchmark figures are generated into the `figs/` directory by the example scripts. These include:

- aNEES and NEES performance plots
- RMSE trajectories

## Dependencies

Developed with tools from the [DECAR group](https://www.decar.ca/):
- [`navlie`](https://github.com/decargroup/navlie) — State estimation tools,
- [`pymlg`](https://github.com/decargroup/pymlg) — Lie group math.

Other library dependencies:
- `numpy`, `scipy`, `matplotlib`, `seaborn`.

See `pyproject.toml` for full dependency list.