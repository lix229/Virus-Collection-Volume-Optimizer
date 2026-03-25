from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_physical_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert RH to fraction and Celsius temperatures to Kelvin."""

    normalized = df.copy()

    if "RH" in normalized.columns:
        rh = pd.to_numeric(normalized["RH"], errors="coerce")
        if rh.max(skipna=True) > 1.0:
            rh = rh / 100.0
        normalized["RH"] = rh

    for column in ("Temperature", "Mod Temp", "Nozzle temp"):
        if column in normalized.columns:
            values = pd.to_numeric(normalized[column], errors="coerce")
            if values.max(skipna=True) < 150:
                values = values + 273.15
            normalized[column] = values

    if "Nozzle temp" in normalized.columns and "Mod Temp" in normalized.columns:
        normalized["(Tnoz-Tmod)"] = normalized["Nozzle temp"] - normalized["Mod Temp"]

    return normalized


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the inference-time feature set expected by the bundled model."""

    engineered = df.copy()
    engineered["collection_medium"] = (
        pd.to_numeric(engineered.get("collection_medium", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    engineered["log_particles"] = np.log1p(engineered["Particles"])
    engineered["particles_temp_interaction"] = (
        engineered["Particles"] * engineered["(Tnoz-Tmod)"]
    )
    engineered["avl_particles"] = engineered["collection_medium"] * engineered["Particles"]
    engineered["avl_temp_diff"] = engineered["collection_medium"] * engineered["(Tnoz-Tmod)"]
    engineered["Mod_x_Temp_x_Nozzle_x_temp"] = (
        engineered["Mod Temp"] * engineered["Nozzle temp"]
    )

    engineered.replace([np.inf, -np.inf], 0.0, inplace=True)
    engineered.fillna(0.0, inplace=True)
    return engineered


def inverse_transform_target(y_transformed: np.ndarray | pd.Series) -> np.ndarray:
    """Map log1p model outputs back to mL."""

    return np.expm1(y_transformed)
