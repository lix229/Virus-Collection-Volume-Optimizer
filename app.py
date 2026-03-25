from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
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
  --bg-main: #445766;
  --bg-shell: #344754;
  --bg-panel: #2a3b49;
  --bg-panel-elevated: #213242;
  --bg-metric: #263745;
  --text-primary: #e8f1f8;
  --text-secondary: #aebfcc;
  --text-muted: #8ea5b6;
  --accent: #8fd3ff;
  --accent-deep: #4e9fd1;
  --border-soft: #4d6a80;
  --border-strong: #6f8ca2;
}
.stApp {
  background:
    radial-gradient(circle at top left, rgba(143, 211, 255, 0.12), transparent 28%),
    linear-gradient(180deg, #4a5d6c 0%, #425563 36%, #394b59 100%);
}
html, body, [class*="css"] {
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  color: var(--text-primary);
}
h1, h2, h3, [data-testid="stMetricLabel"] {
  font-family: "Avenir Next Condensed", "Segoe UI Semibold", sans-serif;
  color: var(--text-primary) !important;
  letter-spacing: 0.02em;
}
p, li, label, div[data-testid="stMarkdownContainer"], div[data-testid="stText"] {
  color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
  color: var(--text-primary) !important;
}
.block-container {
  max-width: 1440px;
  padding-top: 1.15rem;
  padding-bottom: 2rem;
}
.workspace-shell {
  border: 1px solid rgba(143, 211, 255, 0.14);
  background: linear-gradient(180deg, rgba(52, 71, 84, 0.96), rgba(38, 54, 66, 0.96));
  border-radius: 24px;
  padding: 1.05rem 1.2rem 1.15rem 1.2rem;
  box-shadow: 0 18px 40px rgba(13, 22, 30, 0.28);
  margin-bottom: 0.95rem;
}
.workspace-kicker {
  font-size: 0.76rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.13em;
  color: var(--accent) !important;
  margin-bottom: 0.35rem;
}
.workspace-title {
  margin: 0;
  font-size: 2rem;
  line-height: 1.02;
}
.workspace-subtitle {
  margin: 0.4rem 0 0 0;
  max-width: 75rem;
  color: var(--text-secondary) !important;
}
.section-title {
  margin: 0.15rem 0 0.55rem 0;
  font-size: 0.88rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent) !important;
}
.microcopy {
  margin: 0.1rem 0 0.65rem 0;
  font-size: 0.86rem;
  color: var(--text-muted) !important;
}
.control-card {
  border: 1px solid rgba(143, 211, 255, 0.12);
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(42, 59, 73, 0.97), rgba(33, 50, 66, 0.97));
  box-shadow: 0 12px 30px rgba(11, 20, 28, 0.22);
  padding: 0.9rem 1rem 0.35rem 1rem;
  margin-bottom: 0.85rem;
}
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(38, 55, 69, 0.98), rgba(31, 46, 58, 0.98));
  border: 1px solid rgba(143, 211, 255, 0.10);
  border-radius: 16px;
  padding: 0.85rem 1rem;
  box-shadow: 0 10px 20px rgba(10, 18, 25, 0.18);
}
[data-testid="stMetricLabel"] {
  color: var(--text-secondary) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMarkdownContainer"] code,
.stCaption {
  color: var(--text-secondary) !important;
}
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background: #213242 !important;
  color: #e8f1f8 !important;
  border: 1px solid #4d6a80 !important;
  border-radius: 12px !important;
}
div[data-testid="stNumberInput"] input::placeholder {
  color: #8ea5b6 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
  border: 1px solid #8fd3ff !important;
  box-shadow: 0 0 0 1px #8fd3ff !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="input"] input {
  color: #e8f1f8 !important;
}
[data-testid="stDataFrame"] {
  border: 1px solid rgba(143, 211, 255, 0.12);
  border-radius: 16px;
  overflow: hidden;
}
[data-testid="stAlert"] {
  background: rgba(33, 50, 66, 0.92);
  color: var(--text-primary) !important;
  border: 1px solid rgba(143, 211, 255, 0.12);
}
.stDownloadButton button {
  background: linear-gradient(180deg, #3d5f74, #2e4d61);
  color: var(--text-primary);
  border: 1px solid rgba(143, 211, 255, 0.20);
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


def _section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def _apply_plot_theme(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(33, 50, 66, 0.88)",
        plot_bgcolor="rgba(42, 59, 73, 0.94)",
        font_color="#e8f1f8",
        title_font_color="#e8f1f8",
        coloraxis_colorbar=dict(title_font_color="#e8f1f8", tickfont_color="#c7d7e4"),
        xaxis=dict(gridcolor="rgba(143, 211, 255, 0.12)", zerolinecolor="rgba(143, 211, 255, 0.12)"),
        yaxis=dict(gridcolor="rgba(143, 211, 255, 0.12)", zerolinecolor="rgba(143, 211, 255, 0.12)"),
    )


def _render_workspace_header() -> None:
    st.markdown(
        """
<div class="workspace-shell">
  <div class="workspace-kicker">Live Tuning Workspace</div>
  <h1 class="workspace-title">Virus Collection Volume Optimizer</h1>
  <p class="workspace-subtitle">
    Tune environmental and search parameters directly. Rankings, charts, and downloads update automatically,
    and the optimizer always enforces <strong>Nozzle temp &gt;= Mod Temp</strong>.
  </p>
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
- `Mod Temp` / `Nozzle temp`: Search dimensions the optimizer sweeps to find settings close to your selected target.
- `Abs Error (mL)`: Absolute difference from target volume. Lower is better.
            """
        )


def _render_confidence_panel(metadata: dict) -> None:
    metrics = metadata.get("metrics", {})
    validation = metadata.get("validation_metrics", {})
    validation_mape = validation.get("mape")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV RMSE", f"{metrics.get('rmse', float('nan')):.3f}" if metrics else "N/A")
    c2.metric("CV R²", f"{metrics.get('r2', float('nan')):.3f}" if metrics else "N/A")
    c3.metric("Validation MAPE", f"{validation_mape:.1f}%" if validation_mape is not None else "N/A")


def _render_heatmap(results_df: pd.DataFrame, target_volume_ml: float) -> None:
    heat = results_df.pivot(
        index="Mod Temp (C)",
        columns="Nozzle temp (C)",
        values="Predicted Volume (mL)",
    )
    fig = px.imshow(
        heat.sort_index(ascending=True),
        labels={
            "x": "Nozzle temp (C)",
            "y": "Mod Temp (C)",
            "color": "Predicted Volume (mL)",
        },
        color_continuous_scale=[
            [0.0, "#203543"],
            [0.35, "#36566b"],
            [0.65, "#5fa6d1"],
            [1.0, "#a3e0ff"],
        ],
        aspect="auto",
        title=f"Predicted End Volume Across Temperature Combinations (Target = {target_volume_ml:.2f} mL)",
    )
    _apply_plot_theme(fig)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True)


def _render_ranking_plot(results_df: pd.DataFrame) -> None:
    top_n = min(40, len(results_df))
    ranked = results_df.head(top_n).copy()
    ranked["Rank"] = np.arange(1, top_n + 1)
    fig = px.scatter(
        ranked,
        x="Rank",
        y="Abs Error (mL)",
        color="Predicted Volume (mL)",
        size="Predicted Volume (mL)",
        color_continuous_scale=[
            [0.0, "#2f4758"],
            [0.5, "#5fa6d1"],
            [1.0, "#a3e0ff"],
        ],
        hover_data=["Mod Temp (C)", "Nozzle temp (C)"],
        title=f"Top {top_n} Candidates Ranked by Absolute Error",
    )
    _apply_plot_theme(fig)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(fig, use_container_width=True)


def _render_sensitivity(results_df: pd.DataFrame) -> None:
    mod_sensitivity = (
        results_df.groupby("Mod Temp (C)", as_index=False)["Predicted Volume (mL)"].mean().sort_values("Mod Temp (C)")
    )
    nozzle_sensitivity = (
        results_df.groupby("Nozzle temp (C)", as_index=False)["Predicted Volume (mL)"]
        .mean()
        .sort_values("Nozzle temp (C)")
    )

    left, right = st.columns(2)
    with left:
        fig_mod = px.line(
            mod_sensitivity,
            x="Mod Temp (C)",
            y="Predicted Volume (mL)",
            markers=True,
            title="Sensitivity: Avg Predicted Volume vs Mod Temp",
        )
        fig_mod.update_traces(line_color="#8fd3ff", marker_color="#8fd3ff")
        _apply_plot_theme(fig_mod)
        fig_mod.update_layout(margin=dict(l=8, r=8, t=46, b=8))
        st.plotly_chart(fig_mod, use_container_width=True)
    with right:
        fig_nozzle = px.line(
            nozzle_sensitivity,
            x="Nozzle temp (C)",
            y="Predicted Volume (mL)",
            markers=True,
            title="Sensitivity: Avg Predicted Volume vs Nozzle Temp",
        )
        fig_nozzle.update_traces(line_color="#5fa6d1", marker_color="#5fa6d1")
        _apply_plot_theme(fig_nozzle)
        fig_nozzle.update_layout(margin=dict(l=8, r=8, t=46, b=8))
        st.plotly_chart(fig_nozzle, use_container_width=True)


def main() -> None:
    _inject_styles()
    _, _, metadata = _cached_model_artifacts()
    _render_workspace_header()

    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    _section_title("Environmental Inputs")
    st.markdown(
        '<p class="microcopy">These values stay fixed while the optimizer sweeps the moderator and nozzle temperatures.</p>',
        unsafe_allow_html=True,
    )
    env1, env2, env3, env4 = st.columns(4)
    rh_percent = env1.number_input(
        "RH (%)",
        min_value=10.0,
        max_value=95.0,
        value=38.5,
        step=0.1,
        help="Fixed relative humidity during optimization.",
    )
    ambient_temp_c = env2.number_input(
        "Ambient Temperature (C)",
        min_value=10.0,
        max_value=45.0,
        value=23.5,
        step=0.1,
        help="Ambient temperature held constant for all tested combinations.",
    )
    particles = env3.number_input(
        "Particles",
        min_value=1.0,
        max_value=100000.0,
        value=2500.0,
        step=100.0,
        help="Particle concentration/level used as a fixed environmental input.",
    )
    medium_label = env4.selectbox(
        "Collection Medium",
        options=["DI Water (0)", "AVL (1)"],
        index=0,
        help="Encoded as 0 for DI water and 1 for AVL.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    _section_title("Search Grid")
    st.markdown(
        '<p class="microcopy">Results update live. The search always discards any combination where nozzle temperature is below mod temperature.</p>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
    mod_min = s1.number_input("Mod Temp Min (C)", min_value=5.0, max_value=60.0, value=18.0, step=0.5)
    mod_max = s2.number_input("Mod Temp Max (C)", min_value=5.0, max_value=60.0, value=29.0, step=0.5)
    noz_min = s3.number_input("Nozzle Temp Min (C)", min_value=5.0, max_value=90.0, value=22.0, step=0.5)
    noz_max = s4.number_input("Nozzle Temp Max (C)", min_value=5.0, max_value=90.0, value=40.0, step=0.5)
    step_size = s5.number_input(
        "Grid Step (C)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Smaller steps increase search resolution and runtime.",
    )
    target_volume_ml = s6.number_input(
        "Target Volume (mL)",
        min_value=0.1,
        max_value=5.0,
        value=float(DEFAULT_TARGET_VOLUME_ML),
        step=0.1,
    )
    s7.metric("Search Points", "Live", help="The search recomputes automatically when any input changes.")
    st.markdown("</div>", unsafe_allow_html=True)

    _section_title("Model Confidence")
    _render_confidence_panel(metadata)

    collection_medium = 0 if medium_label.startswith("DI Water") else 1

    try:
        mod_values = _build_temperature_values(mod_min, mod_max, step_size)
        nozzle_values = _build_temperature_values(noz_min, noz_max, step_size)
    except ValueError as exc:
        st.error(str(exc))
        return

    full_results = _cached_grid_search(
        rh_percent=float(rh_percent),
        ambient_temp_c=float(ambient_temp_c),
        particles=float(particles),
        collection_medium=int(collection_medium),
        mod_values=tuple(mod_values),
        nozzle_values=tuple(nozzle_values),
        target_volume_ml=float(target_volume_ml),
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

    _render_feature_guide()
    _section_title("Best Candidate")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Mod Temp", f"{best['Mod Temp (C)']:.2f} C")
    m2.metric("Best Nozzle Temp", f"{best['Nozzle temp (C)']:.2f} C")
    m3.metric("Predicted End Volume", f"{best['Predicted Volume (mL)']:.3f} mL")
    m4.metric("Absolute Error", f"{best['Abs Error (mL)']:.3f} mL")

    if "Est. Low (mL)" in results_df.columns and "Est. High (mL)" in results_df.columns:
        st.caption(
            f"Estimated prediction band for best setting (using validation MAPE): "
            f"{best['Est. Low (mL)']:.3f} to {best['Est. High (mL)']:.3f} mL."
        )

    top_rows = min(50, len(results_df))
    upper_left, upper_right = st.columns(2)
    with upper_left:
        _section_title("Ranked Candidates")
        st.dataframe(results_df.head(top_rows), use_container_width=True, hide_index=True, height=500)
    with upper_right:
        _section_title("Heatmap")
        _render_heatmap(results_df, float(target_volume_ml))

    lower_left, lower_right = st.columns(2)
    with lower_left:
        _section_title("Ranking View")
        _render_ranking_plot(results_df)
    with lower_right:
        _section_title("Sensitivity")
        _render_sensitivity(results_df)

    input_summary = {
        "RH (%)": float(rh_percent),
        "Ambient Temperature (C)": float(ambient_temp_c),
        "Particles": float(particles),
        "Collection Medium": medium_label,
        "Mod Temp Range (C)": f"{float(mod_min):.2f} to {float(mod_max):.2f}",
        "Nozzle Temp Range (C)": f"{float(noz_min):.2f} to {float(noz_max):.2f}",
        "Grid Step (C)": float(step_size),
        "Target Volume (mL)": float(target_volume_ml),
    }
    report_markdown = build_markdown_report(
        inputs=input_summary,
        top_results=results_df,
        model_name=metadata.get("model_name"),
        validation_mape=validation_mape,
    )

    _section_title("Download Outputs")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download Ranked Results (CSV)",
            data=results_to_csv_bytes(results_df),
            file_name="optimized_temperature_grid.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download Top 50 (CSV)",
            data=results_to_csv_bytes(results_df.head(50)),
            file_name="optimized_top_50.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Download Optimization Report (MD)",
            data=report_markdown.encode("utf-8"),
            file_name="virus_collection_optimization_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.info(
        "Guidance: prioritize low absolute error first, then favor settings that remain stable across nearby "
        "temperatures in the sensitivity views for easier operational control."
    )


if __name__ == "__main__":
    main()
