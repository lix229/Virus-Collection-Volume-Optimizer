# Virus Collection Volume Optimizer

This directory contains a self-contained Streamlit application for exploring temperature settings that optimize predicted virus collection end volume under fixed environmental conditions.

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

Python dependencies are listed in this directory's `requirements.txt`.

## Installation

If you are working from the parent project directory (`v2`), install the app dependencies with:

```bash
python -m pip install -r streamlit_app/requirements.txt
```

If `streamlit_app` is your current directory or deployment repository root, install with:

```bash
python -m pip install -r requirements.txt
```

## Running the App

From the parent project directory (`v2`), start the Streamlit server with:

```bash
python -m streamlit run streamlit_app/app.py
```

From inside `streamlit_app`, start it with:

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`.

## Deployment

Deploy the app with this same requirements file.

- If deploying the parent project repository, set the requirements file to `streamlit_app/requirements.txt` and the Streamlit entrypoint to `streamlit_app/app.py`.
- If deploying `streamlit_app` as its own repository or app root, set the requirements file to `requirements.txt` and the entrypoint to `app.py`.

## Using the App

1. Set the environmental inputs:
   `RH (%)`, `Ambient Temperature (C)`, `Particles`, and `Collection Medium`
2. Set the search grid:
   `Mod Temp Min/Max`, `Nozzle Temp Min/Max`, `Grid Step (C)`, and `Target Volume (mL)`
3. Review the live-updating outputs:
   best candidate metrics, ranked table, heatmap, ranking plot, and sensitivity plots
4. Download the current results if needed

## Testing

From inside `streamlit_app`, run the included checks with:

```bash
python -m unittest tests.test_app_logic -v
python -m unittest tests.test_standalone_runtime -v
python -m unittest tests.test_sampler_optimizer -v
```

From the parent project directory (`v2`), either change into the app directory first or run:

```bash
cd streamlit_app
python -m unittest tests.test_app_logic -v
python -m unittest tests.test_standalone_runtime -v
python -m unittest tests.test_sampler_optimizer -v
```

These tests cover:

- core app helper behavior
- standalone runtime behavior when the app directory is copied and executed independently
