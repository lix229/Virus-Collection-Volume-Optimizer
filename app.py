from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .sampler_optimizer import (
        DEFAULT_TARGET_VOLUME_ML,
        build_markdown_report,
        load_model_artifacts,
        optimize_temperature_grid,
        results_to_csv_bytes,
    )
except ImportError:
    from sampler_optimizer import (
        DEFAULT_TARGET_VOLUME_ML,
        build_markdown_report,
        load_model_artifacts,
        optimize_temperature_grid,
        results_to_csv_bytes,
    )


st.set_page_config(
    page_title="Virus Collection Volume Optimizer",
    page_icon=":test_tube:",
    layout="wide",
)


def _build_theme_css() -> str:
    return """
<style>
:root {
  --page-bg: #F3F6F8;
  --panel: #FFFFFF;
  --panel-soft: #F8FAFB;
  --ink: #24313A;
  --muted: #63717B;
  --line: #DDE5EA;
  --input-line: #CCD8DE;
  --primary: #2D6F73;
  --primary-hover: #245C60;
  --primary-soft: #E6F1F1;
  --secondary: #496F8F;
  --secondary-soft: #E8F0F6;
  --warning: #9B6A2F;
  --warning-soft: #F7EEDD;
  --success: #3F7A5F;
  --success-soft: #E7F2EC;
  --table-header: #EEF4F6;
  --table-row: #FFFFFF;
  --table-alt: #F7FAFB;
  --table-hover: #EDF6F6;
  --shadow: 0 10px 24px rgba(36, 49, 58, 0.07);
}
.stApp {
  background: var(--page-bg);
}
html, body, [class*="css"] {
  font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
  color: var(--ink);
}
h1, h2, h3, [data-testid="stMetricLabel"] {
  font-family: "Avenir Next", "Segoe UI Semibold", Arial, sans-serif;
  color: var(--ink) !important;
  letter-spacing: 0;
}
p, li, label, div[data-testid="stMarkdownContainer"], div[data-testid="stText"] {
  color: var(--muted) !important;
}
.block-container {
  max-width: 1760px;
  padding-top: 2.75rem;
  padding-left: 1rem;
  padding-right: 1rem;
  padding-bottom: 2rem;
}
.workspace-shell {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.75rem;
  box-shadow: var(--shadow);
}
.workspace-kicker {
  font-size: 0.75rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--primary) !important;
  margin-bottom: 0.25rem;
}
.workspace-title {
  margin: 0;
  font-size: 1.85rem !important;
  line-height: 1.12 !important;
}
.workspace-subtitle {
  margin: 0.35rem 0 0 0;
  max-width: 78rem;
  color: var(--muted) !important;
}
.model-pill {
  display: inline-block;
  border: 1px solid #B9D1D1;
  background: var(--primary-soft);
  color: #214D50 !important;
  border-radius: 999px;
  padding: 0.24rem 0.62rem;
  margin-top: 0.65rem;
  margin-right: 0.35rem;
  font-size: 0.78rem;
  font-weight: 800;
}
.section-title {
  margin: 0.2rem 0 0.55rem 0;
  font-size: 0.8rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--primary) !important;
}
.microcopy {
  margin: 0.1rem 0 0.65rem 0;
  font-size: 0.86rem;
  color: var(--muted) !important;
}
.input-panel-title {
  margin: 0 0 0.65rem 0;
  font-size: 1.18rem;
  line-height: 1.2;
  font-weight: 800;
  color: var(--ink) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel-soft);
  border-color: var(--line) !important;
  border-radius: 8px;
  box-shadow: 0 8px 20px rgba(36, 49, 58, 0.04);
}
.workbench-note {
  border: 1px solid #E5D4AA;
  background: var(--warning-soft);
  border-radius: 8px;
  padding: 0.75rem 0.85rem;
  color: #60451D !important;
  font-size: 0.84rem;
  margin: 0.65rem 0 0.8rem;
}
.stable-zone {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.75rem;
  align-items: center;
  border: 1px solid #B9D1D1;
  background: var(--primary-soft);
  border-radius: 8px;
  padding: 0.75rem 0.85rem;
  margin-top: 0.75rem;
  color: #214D50 !important;
}
.stable-zone strong {
  display: block;
  color: var(--ink) !important;
  margin-bottom: 0.15rem;
}
.stable-zone span {
  color: #214D50 !important;
}
.stable-zone-badge {
  border: 1px solid #B9D1D1;
  border-radius: 999px;
  background: var(--panel);
  color: #214D50 !important;
  padding: 0.2rem 0.55rem;
  font-size: 0.76rem;
  font-weight: 800;
  white-space: nowrap;
}
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  box-shadow: 0 6px 16px rgba(36, 49, 58, 0.05);
}
[data-testid="stMetricValue"] {
  color: var(--ink) !important;
}
[data-testid="stMetricValue"] > div {
  font-size: clamp(1.55rem, 2.1vw, 2rem) !important;
  line-height: 1.15 !important;
  white-space: normal !important;
}
[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
}
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background: #ffffff !important;
  color: var(--ink) !important;
  border: 1px solid var(--input-line) !important;
  border-radius: 6px !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
  border: 1px solid var(--primary) !important;
  box-shadow: 0 0 0 1px var(--primary) !important;
}
[data-testid="stPlotlyChart"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.35rem;
}
[data-testid="stAlert"] {
  background: var(--panel);
  color: var(--ink) !important;
  border: 1px solid var(--line);
}
.stDownloadButton button,
.stButton button {
  border-radius: 6px;
  border: 1px solid var(--line);
  font-weight: 800;
  background: var(--panel) !important;
  color: var(--ink) !important;
}
.stDownloadButton button p,
.stDownloadButton button span,
.stButton button p,
.stButton button span {
  color: inherit !important;
}
.stDownloadButton button {
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border-color: var(--primary) !important;
}
.stDownloadButton button div[data-testid="stMarkdownContainer"],
.stDownloadButton button div[data-testid="stMarkdownContainer"] p,
.stDownloadButton button div[data-testid="stMarkdownContainer"] span {
  color: #FFFFFF !important;
}
.stDownloadButton button:hover {
  background: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
}
[data-baseweb="popover"] [role="listbox"],
[role="listbox"] {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  color: var(--ink) !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] ul {
  background: var(--panel) !important;
  color: var(--ink) !important;
}
[role="option"],
[role="option"] * {
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
}
[role="option"]:not(:hover):not([aria-selected="true"]),
[role="option"]:not(:hover):not([aria-selected="true"]) > div {
  background: var(--panel) !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
  background: var(--primary-soft) !important;
}
button[role="tab"] p {
  color: var(--muted) !important;
}
button[role="tab"][aria-selected="true"] p {
  color: var(--primary) !important;
}
[data-baseweb="tab-highlight"] {
  background-color: var(--primary) !important;
}
[data-baseweb="tab-border"] {
  background-color: var(--line) !important;
}
.results-table-wrap {
  width: 100%;
  max-width: 100%;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--table-row);
}
.results-table {
  width: 100%;
  min-width: var(--results-table-min-width, 100%);
  border-collapse: separate;
  border-spacing: 0;
  color: var(--ink) !important;
  font-size: 0.82rem;
}
.results-table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--table-header);
  color: var(--ink) !important;
  font-weight: 800;
  text-align: right;
  border-bottom: 1px solid var(--line);
  padding: 0.62rem 0.7rem;
  white-space: nowrap;
}
.results-table tbody td {
  background: var(--table-row);
  color: var(--ink) !important;
  text-align: right;
  border-bottom: 1px solid var(--line);
  padding: 0.55rem 0.7rem;
  white-space: nowrap;
}
.results-table tbody tr:nth-child(even) td {
  background: var(--table-alt);
}
.results-table tbody tr:hover td {
  background: var(--table-hover);
}
.results-table tbody tr:last-child td {
  border-bottom: 0;
}
@media (max-width: 720px) {
  .stable-zone {
    grid-template-columns: 1fr;
  }
}
</style>
    """


def _inject_styles() -> None:
    st.markdown(_build_theme_css(), unsafe_allow_html=True)


def _build_temperature_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be greater than 0.")
    if stop < start:
        raise ValueError("Maximum temperature must be greater than or equal to minimum temperature.")

    total_steps = int(np.floor((stop - start) / step))
    values = [round(start + i * step, 4) for i in range(total_steps + 1)]
    if values[-1] < stop - 1e-9:
        values.append(round(stop, 4))
    return values


def _filter_nozzle_ge_mod_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df[results_df["Nozzle temp (C)"] >= results_df["Mod Temp (C)"]].reset_index(drop=True)


@st.cache_resource
def _cached_model_artifacts():
    return load_model_artifacts("models")


@st.cache_data(show_spinner=False)
def _cached_grid_search(
    *,
    rh_percent: float,
    ambient_temp_c: float,
    particles: float,
    collection_medium: int,
    mod_values: tuple[float, ...],
    nozzle_values: tuple[float, ...],
    target_volume_ml: float,
) -> pd.DataFrame:
    model, feature_columns, _ = _cached_model_artifacts()
    return optimize_temperature_grid(
        model=model,
        feature_columns=feature_columns,
        rh_percent=rh_percent,
        ambient_temp_c=ambient_temp_c,
        particles=particles,
        collection_medium=collection_medium,
        mod_temp_values=mod_values,
        nozzle_temp_values=nozzle_values,
        target_volume_ml=target_volume_ml,
    )


def _apply_prediction_band(df: pd.DataFrame, mape_percent: float | None) -> pd.DataFrame:
    if mape_percent is None:
        return df
    mape_ratio = float(mape_percent) / 100.0
    out = df.copy()
    out["Est. Low (mL)"] = (out["Predicted Volume (mL)"] * (1 - mape_ratio)).clip(lower=0.0)
    out["Est. High (mL)"] = out["Predicted Volume (mL)"] * (1 + mape_ratio)
    return out


def _calculate_local_stability(results_df: pd.DataFrame, best_row: pd.Series) -> tuple[str, float, int]:
    nearby = results_df[
        (results_df["Mod Temp (C)"].sub(float(best_row["Mod Temp (C)"])).abs() <= 1.0)
        & (results_df["Nozzle temp (C)"].sub(float(best_row["Nozzle temp (C)"])).abs() <= 1.0)
    ]
    if nearby.empty:
        nearby = results_df.head(min(5, len(results_df)))

    spread = float(nearby["Predicted Volume (mL)"].max() - nearby["Predicted Volume (mL)"].min())
    if spread <= 0.05:
        label = "High"
    elif spread <= 0.15:
        label = "Moderate"
    else:
        label = "Variable"
    return label, spread, len(nearby)


def _section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def _apply_plot_theme(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#FFFFFF",
        font_color="#24313A",
        title_font_color="#24313A",
        title_text="",
        coloraxis_colorbar=dict(title_font_color="#24313A", tickfont_color="#63717B"),
        legend=dict(font=dict(color="#24313A"), bgcolor="rgba(255, 255, 255, 0)"),
        xaxis=dict(
            gridcolor="#E7EDF1",
            zerolinecolor="#DDE5EA",
            tickfont=dict(color="#63717B"),
            title_font=dict(color="#24313A"),
        ),
        yaxis=dict(
            gridcolor="#E7EDF1",
            zerolinecolor="#DDE5EA",
            tickfont=dict(color="#63717B"),
            title_font=dict(color="#24313A"),
        ),
    )
    fig.update_xaxes(tickfont=dict(color="#63717B"), title_font=dict(color="#24313A"))
    fig.update_yaxes(tickfont=dict(color="#63717B"), title_font=dict(color="#24313A"))


def _format_temperature(value: float) -> str:
    rounded = round(float(value), 1)
    return f"{rounded:.0f}" if rounded.is_integer() else f"{rounded:.1f}"


def _render_workspace_header(metadata: dict[str, Any]) -> None:
    model_name = metadata.get("model_name") or "Unknown model"
    validation_mape = metadata.get("validation_metrics", {}).get("mape")
    mape_label = f"Validation MAPE {validation_mape:.1f}%" if validation_mape is not None else "Validation MAPE unavailable"
    st.markdown(
        f"""
<div class="workspace-shell">
  <div class="workspace-kicker">Live Tuning Workspace</div>
  <h1 class="workspace-title">Virus Collection Volume Optimizer</h1>
  <p class="workspace-subtitle">
    Set the environmental inputs once, scan valid temperature combinations, and review the best setting with
    local stability before exporting results.
  </p>
  <span class="model-pill">Model: {model_name}</span>
  <span class="model-pill">{mape_label}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_feature_guide() -> None:
    with st.expander("Feature Guide and Interpretation", expanded=False):
        st.markdown(
            """
- `RH (%)`: Relative humidity. Higher RH can reduce evaporation and preserve collection volume.
- `Ambient Temperature (C)`: Lab air temperature surrounding the sampler.
- `Particles`: Particle concentration loaded into collection conditions.
- `Collection Medium`: `DI Water` (`0`) or `AVL` (`1`) as encoded in training.
- `Target Volume (mL)`: Desired end volume used for ranking settings.
- `Mod Temp` / `Nozzle temp`: Search dimensions swept by the optimizer.
- `Abs Error (mL)`: Absolute difference from target volume. Lower is better.
            """
        )


def _render_confidence_panel(metadata: dict[str, Any]) -> None:
    metrics = metadata.get("metrics", {})
    validation = metadata.get("validation_metrics", {})
    validation_mape = validation.get("mape")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV RMSE", f"{metrics.get('rmse', float('nan')):.3f}" if metrics else "N/A")
    c2.metric("CV R2", f"{metrics.get('r2', float('nan')):.3f}" if metrics else "N/A")
    c3.metric("Validation MAPE", f"{validation_mape:.1f}%" if validation_mape is not None else "N/A")


def _render_input_controls() -> dict[str, Any]:
    with st.container(border=True):
        st.markdown('<div class="input-panel-title">Inputs</div>', unsafe_allow_html=True)
        _section_title("Environment")
        rh_percent = st.number_input(
            "RH (%)",
            min_value=10.0,
            max_value=95.0,
            value=38.5,
            step=0.1,
            help="Fixed relative humidity during optimization.",
        )
        ambient_temp_c = st.number_input(
            "Ambient Temperature (C)",
            min_value=10.0,
            max_value=45.0,
            value=23.5,
            step=0.1,
            help="Ambient temperature held constant for all tested combinations.",
        )
        particles = st.number_input(
            "Particles",
            min_value=1.0,
            max_value=100000.0,
            value=2500.0,
            step=100.0,
            help="Particle concentration/level used as a fixed environmental input.",
        )
        medium_label = st.selectbox(
            "Collection Medium",
            options=["DI Water (0)", "AVL (1)"],
            index=0,
            help="Encoded as 0 for DI water and 1 for AVL.",
        )

        _section_title("Temperature Search")
        mod_min = st.number_input("Mod Temp Min (C)", min_value=5.0, max_value=60.0, value=18.0, step=0.5)
        mod_max = st.number_input("Mod Temp Max (C)", min_value=5.0, max_value=60.0, value=29.0, step=0.5)
        noz_min = st.number_input("Nozzle Temp Min (C)", min_value=5.0, max_value=90.0, value=22.0, step=0.5)
        noz_max = st.number_input("Nozzle Temp Max (C)", min_value=5.0, max_value=90.0, value=40.0, step=0.5)
        step_size = st.number_input(
            "Grid Step (C)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Smaller steps increase search resolution and runtime.",
        )
        target_volume_ml = st.number_input(
            "Target Volume (mL)",
            min_value=0.1,
            max_value=5.0,
            value=float(DEFAULT_TARGET_VOLUME_ML),
            step=0.1,
        )
        st.markdown(
            '<div class="workbench-note">Constraint active: nozzle temperature must be greater than or equal to moderator temperature.</div>',
            unsafe_allow_html=True,
        )

    return {
        "rh_percent": float(rh_percent),
        "ambient_temp_c": float(ambient_temp_c),
        "particles": float(particles),
        "medium_label": medium_label,
        "collection_medium": 0 if medium_label.startswith("DI Water") else 1,
        "mod_min": float(mod_min),
        "mod_max": float(mod_max),
        "noz_min": float(noz_min),
        "noz_max": float(noz_max),
        "step_size": float(step_size),
        "target_volume_ml": float(target_volume_ml),
    }


def _render_heatmap(results_df: pd.DataFrame, target_volume_ml: float) -> None:
    heat = results_df.pivot(
        index="Mod Temp (C)",
        columns="Nozzle temp (C)",
        values="Predicted Volume (mL)",
    ).sort_index(ascending=True)
    heat = heat.reindex(sorted(heat.columns), axis=1)
    fig = go.Figure(
        data=go.Heatmap(
            x=heat.columns,
            y=heat.index,
            z=heat.values,
            colorscale=[
                [0.0, "#DDEDEA"],
                [0.35, "#9ACAC7"],
                [0.65, "#2D6F73"],
                [0.82, "#E8C77A"],
                [1.0, "#B9853D"],
            ],
            colorbar=dict(title="Predicted Volume (mL)", thickness=14),
            hovertemplate="Mod Temp: %{y:.1f} C<br>Nozzle Temp: %{x:.1f} C<br>Predicted: %{z:.3f} mL<extra></extra>",
        )
    )
    _apply_plot_theme(fig)
    fig.add_contour(
        x=heat.columns,
        y=heat.index,
        z=np.abs(heat.values - target_volume_ml),
        contours=dict(start=0, end=0.05, size=0.05, coloring="none", showlabels=False),
        line=dict(color="#245C60", width=1),
        showscale=False,
        hoverinfo="skip",
    )
    fig.update_layout(height=560, margin=dict(l=18, r=18, t=10, b=8))
    fig.update_xaxes(title_text="Nozzle temp (C)", side="bottom")
    fig.update_yaxes(title_text="Mod Temp (C)", autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_stable_zone_note(best: pd.Series, stability_count: int) -> None:
    st.markdown(
        f"""
<div class="stable-zone">
  <span>
    <strong>Best zone spans adjacent settings</strong>
    The selected setting is compared against {stability_count} nearby valid combinations around
    {float(best["Mod Temp (C)"]):.1f} C mod temp and {float(best["Nozzle temp (C)"]):.1f} C nozzle temp.
  </span>
  <span class="stable-zone-badge">Stable zone</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_stability_preview(results_df: pd.DataFrame, target_volume_ml: float) -> None:
    mod_sensitivity = (
        results_df.groupby("Mod Temp (C)", as_index=False)["Predicted Volume (mL)"].mean().sort_values("Mod Temp (C)")
    )
    nozzle_sensitivity = (
        results_df.groupby("Nozzle temp (C)", as_index=False)["Predicted Volume (mL)"]
        .mean()
        .sort_values("Nozzle temp (C)")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mod_sensitivity["Mod Temp (C)"],
            y=mod_sensitivity["Predicted Volume (mL)"],
            mode="lines+markers",
            name="Mod Temp average",
            line=dict(color="#2D6F73", width=3),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nozzle_sensitivity["Nozzle temp (C)"],
            y=nozzle_sensitivity["Predicted Volume (mL)"],
            mode="lines+markers",
            name="Nozzle Temp average",
            line=dict(color="#496F8F", width=3),
            marker=dict(size=6),
        )
    )
    fig.add_hline(
        y=target_volume_ml,
        line_color="#9B6A2F",
        line_dash="dash",
        annotation_text="Target",
        annotation_position="top left",
    )
    _apply_plot_theme(fig)
    fig.update_layout(
        height=285,
        margin=dict(l=8, r=8, t=16, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="left",
            x=0,
            font=dict(color="#24313A"),
            bgcolor="rgba(255, 255, 255, 0)",
        ),
        xaxis_title="Temperature (C)",
        yaxis_title="Avg predicted volume (mL)",
    )
    fig.update_xaxes(title_font=dict(color="#24313A"), tickfont=dict(color="#63717B"))
    fig.update_yaxes(title_font=dict(color="#24313A"), tickfont=dict(color="#63717B"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _format_numeric_cell(value: Any, decimals: int) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _table_formatters(df: pd.DataFrame) -> dict[str, Any]:
    formatters: dict[str, Any] = {}
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        decimals = 4
        if "temp" in column.lower() or column in {"RH (%)", "Ambient Temperature (C)", "Mod C", "Nozzle C"}:
            decimals = 1
        elif column == "Particles":
            decimals = 0
        formatters[column] = lambda value, precision=decimals: _format_numeric_cell(value, precision)
    return formatters


def _format_results_table_html(df: pd.DataFrame, max_height_px: int, min_width_px: int | None = None) -> str:
    max_height = max(120, int(max_height_px))
    inline_style = f"max-height: {max_height}px;"
    if min_width_px is not None:
        inline_style += f" --results-table-min-width: {max(320, int(min_width_px))}px;"
    table_html = df.to_html(
        index=False,
        classes="results-table",
        border=0,
        escape=True,
        formatters=_table_formatters(df),
    )
    return f'<div class="results-table-wrap" style="{inline_style}">{table_html}</div>'


def _render_results_table(df: pd.DataFrame, max_height_px: int, min_width_px: int | None = None) -> None:
    st.markdown(
        _format_results_table_html(df, max_height_px=max_height_px, min_width_px=min_width_px),
        unsafe_allow_html=True,
    )


def _render_downloads(results_df: pd.DataFrame, report_markdown: str) -> None:
    _section_title("Download Outputs")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download Ranked Results (CSV)",
            data=results_to_csv_bytes(results_df),
            file_name="optimized_temperature_grid.csv",
            mime="text/csv",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Download Top 50 (CSV)",
            data=results_to_csv_bytes(results_df.head(50)),
            file_name="optimized_top_50.csv",
            mime="text/csv",
            width="stretch",
        )
    with d3:
        st.download_button(
            "Download Optimization Report (MD)",
            data=report_markdown.encode("utf-8"),
            file_name="virus_collection_optimization_report.md",
            mime="text/markdown",
            width="stretch",
        )


def _render_workspace(controls: dict[str, Any], metadata: dict[str, Any]) -> None:
    _render_workspace_header(metadata)

    try:
        mod_values = _build_temperature_values(controls["mod_min"], controls["mod_max"], controls["step_size"])
        nozzle_values = _build_temperature_values(controls["noz_min"], controls["noz_max"], controls["step_size"])
    except ValueError as exc:
        st.error(str(exc))
        return

    full_results = _cached_grid_search(
        rh_percent=controls["rh_percent"],
        ambient_temp_c=controls["ambient_temp_c"],
        particles=controls["particles"],
        collection_medium=int(controls["collection_medium"]),
        mod_values=tuple(mod_values),
        nozzle_values=tuple(nozzle_values),
        target_volume_ml=controls["target_volume_ml"],
    )
    results_df = _filter_nozzle_ge_mod_results(full_results)

    if results_df.empty:
        st.warning("No valid parameter combinations remain after enforcing nozzle temp >= mod temp. Adjust the search ranges.")
        return

    if len(results_df) > 5000:
        st.warning("Large grid detected. Showing the top 5000 combinations sorted by error.")
        results_df = results_df.head(5000)

    validation_mape = metadata.get("validation_metrics", {}).get("mape")
    results_df = _apply_prediction_band(results_df, validation_mape)
    best = results_df.iloc[0]
    stability_label, stability_spread, stability_count = _calculate_local_stability(results_df, best)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Recommended Setting",
        f"{_format_temperature(best['Mod Temp (C)'])}/{_format_temperature(best['Nozzle temp (C)'])} C",
        help="Mod Temp / Nozzle Temp",
    )
    m2.metric("Predicted Volume", f"{best['Predicted Volume (mL)']:.3f} mL", help="Predicted end volume")
    m3.metric("Absolute Error", f"{best['Abs Error (mL)']:.3f} mL", help="Distance from target volume")
    m4.metric("Local Stability", stability_label, help=f"Nearby predicted-volume spread: {stability_spread:.3f} mL")

    if "Est. Low (mL)" in results_df.columns and "Est. High (mL)" in results_df.columns:
        st.caption(
            f"Estimated prediction band for best setting (using validation MAPE): "
            f"{best['Est. Low (mL)']:.3f} to {best['Est. High (mL)']:.3f} mL."
        )

    input_summary = {
        "RH (%)": controls["rh_percent"],
        "Ambient Temperature (C)": controls["ambient_temp_c"],
        "Particles": controls["particles"],
        "Collection Medium": controls["medium_label"],
        "Mod Temp Range (C)": f"{controls['mod_min']:.2f} to {controls['mod_max']:.2f}",
        "Nozzle Temp Range (C)": f"{controls['noz_min']:.2f} to {controls['noz_max']:.2f}",
        "Grid Step (C)": controls["step_size"],
        "Target Volume (mL)": controls["target_volume_ml"],
    }
    report_markdown = build_markdown_report(
        inputs=input_summary,
        top_results=results_df,
        model_name=metadata.get("model_name"),
        validation_mape=validation_mape,
    )

    map_col, decision_col = st.columns([1.1, 0.9], gap="medium")
    with map_col:
        _section_title("Temperature Response Map")
        st.caption("The map remains the primary visual for finding stable near-target zones.")
        _render_heatmap(results_df, float(controls["target_volume_ml"]))
        _render_stable_zone_note(best, stability_count)

    with decision_col:
        decision_tab, table_tab, report_tab = st.tabs(["Decision", "Full Table", "Report"])
        with decision_tab:
            _section_title("Top Candidates")
            st.caption("Keep the table short, then check whether the winner is fragile.")
            top_candidate_columns = [
                "Mod Temp (C)",
                "Nozzle temp (C)",
                "Predicted Volume (mL)",
                "Abs Error (mL)",
            ]
            top_candidates = results_df.loc[:, top_candidate_columns].head(4).rename(
                columns={
                    "Mod Temp (C)": "Mod C",
                    "Nozzle temp (C)": "Nozzle C",
                    "Predicted Volume (mL)": "Volume mL",
                    "Abs Error (mL)": "Error mL",
                }
            )
            _render_results_table(top_candidates, max_height_px=215)
            _section_title("Stability Preview")
            st.caption("Flatter nearby values indicate easier operational control.")
            _render_stability_preview(results_df, float(controls["target_volume_ml"]))
        with table_tab:
            _section_title("Ranked Candidates")
            _render_results_table(results_df.head(50), max_height_px=520, min_width_px=860)
        with report_tab:
            _section_title("Model Confidence")
            _render_confidence_panel(metadata)
            st.markdown(
                '<div class="workbench-note">Prioritize low absolute error first, then favor settings that remain stable across nearby temperatures.</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    _render_downloads(results_df, report_markdown)


def main() -> None:
    _inject_styles()
    _, _, metadata = _cached_model_artifacts()

    input_col, workspace_col = st.columns([0.22, 0.78], gap="medium")
    with input_col:
        controls = _render_input_controls()
        _render_feature_guide()

    with workspace_col:
        _render_workspace(controls, metadata)


if __name__ == "__main__":
    main()
