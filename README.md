# Ocean Current Modeling with Gaussian Processes

Gaussian Process regression and particle-flow simulation on ocean
current data, based on the MITx 6.419x environmental-data module,
reimplemented from scratch and extended.

## Setup
1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Download the flow data into `data/` (see notebooks/01)

## Structure
- `oceangp/` — kernels, GP model, particle simulator
- `notebooks/` — analysis and figures
- `app.py` — interactive demo