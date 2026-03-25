# Virus Collection Volume Optimizer

This repository contains a self-contained Streamlit application for exploring temperature settings that optimize predicted virus collection end volume under fixed environmental conditions.

The app is designed for fast parameter tuning. As inputs change, the ranked results, charts, and downloadable outputs update automatically. The search always enforces `Nozzle temp >= Mod Temp`.

## Overview

The application allows a user to:

- Set fixed environmental inputs such as relative humidity, ambient temperature, particle level, and collection medium
- Sweep moderator and nozzle temperature ranges against a target end volume
- Review the best candidate settings, ranked combinations, heatmap, and sensitivity plots
- Download the full ranked grid, the top 50 results, and a Markdown report

## Repository Contents

- `app.py`
  Streamlit application entrypoint and interface logic
- `sampler_optimizer.py`
  Inference and grid search helpers
- `feature_engineering.py`
  Local preprocessing required for inference
- `models/`
  Bundled trained model artifacts used at runtime
- `tests/`
  Basic app logic and standalone runtime tests

## Requirements

- Python 3.10 or later is recommended
- `pip` for dependency installation

Python dependencies are listed in `requirements.txt`.

## Installation

Clone the repository and install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit server from the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`.

## Using the App

1. Set the environmental inputs:
   `RH (%)`, `Ambient Temperature (C)`, `Particles`, and `Collection Medium`
2. Set the search grid:
   `Mod Temp Min/Max`, `Nozzle Temp Min/Max`, `Grid Step (C)`, and `Target Volume (mL)`
3. Review the live-updating outputs:
   best candidate metrics, ranked table, heatmap, ranking plot, and sensitivity plots
4. Download the current results if needed

## Testing

Run the included checks from the repository root:

```bash
python -m unittest tests.test_app_logic -v
python -m unittest tests.test_standalone_runtime -v
```

These tests cover:

- core app helper behavior
- standalone runtime behavior when the app directory is copied and executed independently
