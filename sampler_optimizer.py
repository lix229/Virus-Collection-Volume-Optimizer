from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd

try:
    from .feature_engineering import engineer_features, inverse_transform_target, normalize_physical_units
except ImportError:
    from feature_engineering import engineer_features, inverse_transform_target, normalize_physical_units


APP_ROOT = Path(__file__).resolve().parent


DEFAULT_TARGET_VOLUME_ML = 1.5


def load_model_artifacts(models_dir: Path | str = "models") -> tuple[Any, list[str], dict[str, Any]]:
    models_path = Path(models_dir)
    if not models_path.is_absolute():
        models_path = APP_ROOT / models_path
    model = joblib.load(models_path / "best_model.joblib")
    feature_columns = json.loads((models_path / "feature_columns.json").read_text(encoding="utf-8"))
    metadata_path = models_path / "best_model_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, feature_columns, metadata


def build_feature_frame(
    *,
    rh_percent: float,
    ambient_temp_c: float,
    particles: float,
    mod_temp_c: float,
    nozzle_temp_c: float,
    collection_medium: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    base_frame = pd.DataFrame(
        [
            {
                "RH": float(rh_percent),
                "Temperature": float(ambient_temp_c),
                "Particles": float(particles),
                "Mod Temp": float(mod_temp_c),
                "Nozzle temp": float(nozzle_temp_c),
                "(Tnoz-Tmod)": float(nozzle_temp_c) - float(mod_temp_c),
                "collection_medium": int(collection_medium),
            }
        ]
    )

    normalized = normalize_physical_units(base_frame)
    normalized["(Tnoz-Tmod)"] = normalized["Nozzle temp"] - normalized["Mod Temp"]
    engineered = engineer_features(normalized)
    return engineered.reindex(columns=feature_columns, fill_value=0.0)


def predict_end_volume_ml(
    *,
    model: Any,
    feature_columns: list[str],
    rh_percent: float,
    ambient_temp_c: float,
    particles: float,
    mod_temp_c: float,
    nozzle_temp_c: float,
    collection_medium: int,
) -> float:
    feature_frame = build_feature_frame(
        rh_percent=rh_percent,
        ambient_temp_c=ambient_temp_c,
        particles=particles,
        mod_temp_c=mod_temp_c,
        nozzle_temp_c=nozzle_temp_c,
        collection_medium=collection_medium,
        feature_columns=feature_columns,
    )
    predicted_log = model.predict(feature_frame)
    predicted_ml = inverse_transform_target(predicted_log)
    return float(predicted_ml[0])


def optimize_temperature_grid(
    *,
    model: Any,
    feature_columns: list[str],
    rh_percent: float,
    ambient_temp_c: float,
    particles: float,
    collection_medium: int,
    mod_temp_values: Iterable[float],
    nozzle_temp_values: Iterable[float],
    target_volume_ml: float = DEFAULT_TARGET_VOLUME_ML,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for mod_temp in mod_temp_values:
        for nozzle_temp in nozzle_temp_values:
            predicted_ml = predict_end_volume_ml(
                model=model,
                feature_columns=feature_columns,
                rh_percent=rh_percent,
                ambient_temp_c=ambient_temp_c,
                particles=particles,
                mod_temp_c=float(mod_temp),
                nozzle_temp_c=float(nozzle_temp),
                collection_medium=collection_medium,
            )
            abs_error = abs(predicted_ml - float(target_volume_ml))
            rows.append(
                {
                    "Mod Temp (C)": float(mod_temp),
                    "Nozzle temp (C)": float(nozzle_temp),
                    "Predicted Volume (mL)": predicted_ml,
                    "Abs Error (mL)": abs_error,
                }
            )

    return pd.DataFrame(rows).sort_values("Abs Error (mL)", ascending=True).reset_index(drop=True)


def results_to_csv_bytes(results: pd.DataFrame) -> bytes:
    return results.to_csv(index=False).encode("utf-8")


def build_markdown_report(
    *,
    inputs: dict[str, Any],
    top_results: pd.DataFrame,
    model_name: str | None,
    validation_mape: float | None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = [
        "# Virus Collection Optimization Report",
        "",
        f"Generated at: {timestamp}",
        "",
        "## Model Information",
        f"- Model: {model_name or 'Unknown'}",
        (
            f"- Validation MAPE: {validation_mape:.1f}%"
            if validation_mape is not None
            else "- Validation MAPE: Not available"
        ),
        "",
        "## Input Summary",
    ]

    for key, value in inputs.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Top Recommendations")
    lines.append("")

    if top_results.empty:
        lines.append("No optimization results available.")
        return "\n".join(lines)

    capped = top_results.head(10).copy()
    selected_cols = [
        col
        for col in [
            "Mod Temp (C)",
            "Nozzle temp (C)",
            "Predicted Volume (mL)",
            "Abs Error (mL)",
            "Est. Low (mL)",
            "Est. High (mL)",
        ]
        if col in capped.columns
    ]

    lines.append("| Rank | " + " | ".join(selected_cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(selected_cols)) + "|")
    for idx, row in capped.iterrows():
        values: list[str] = []
        for col in selected_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + str(int(idx) + 1) + " | " + " | ".join(values) + " |")

    return "\n".join(lines)
