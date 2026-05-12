from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import app
from app import _apply_plot_theme, _build_theme_css, _build_temperature_values, _filter_nozzle_ge_mod_results


APP_ROOT = Path(__file__).resolve().parents[1]


class AppLogicTest(unittest.TestCase):
    def test_theme_css_defines_light_inputs_with_visible_text(self) -> None:
        css = _build_theme_css()

        self.assertIn('div[data-testid="stNumberInput"] input', css)
        self.assertIn("color: var(--ink)", css)
        self.assertIn("background: #ffffff", css)
        self.assertIn("border: 1px solid var(--input-line)", css)
        self.assertIn("box-shadow: 0 0 0 1px var(--primary)", css)

    def test_build_temperature_values_rejects_reversed_range(self) -> None:
        with self.assertRaises(ValueError):
            _build_temperature_values(25.0, 20.0, 1.0)

    def test_filter_nozzle_ge_mod_results_keeps_only_valid_pairs(self) -> None:
        frame = pd.DataFrame(
            [
                {"Mod Temp (C)": 18.0, "Nozzle temp (C)": 22.0, "Predicted Volume (mL)": 1.9},
                {"Mod Temp (C)": 23.0, "Nozzle temp (C)": 22.0, "Predicted Volume (mL)": 1.5},
                {"Mod Temp (C)": 21.0, "Nozzle temp (C)": 21.0, "Predicted Volume (mL)": 1.7},
            ]
        )

        filtered = _filter_nozzle_ge_mod_results(frame)

        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered["Nozzle temp (C)"] >= filtered["Mod Temp (C)"]).all())

    def test_app_uses_current_streamlit_width_api(self) -> None:
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertNotIn("use_container_width", source)
        self.assertIn('width="stretch"', source)

    def test_app_source_contains_a2_workbench_structure(self) -> None:
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        expected_labels = [
            "Recommended Setting",
            "Local Stability",
            "Temperature Response Map",
            "Top Candidates",
            "Stability Preview",
            "Download Outputs",
        ]
        for label in expected_labels:
            with self.subTest(label=label):
                self.assertIn(label, source)

    def test_app_uses_fixed_input_rail_for_primary_controls(self) -> None:
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("_render_input_controls", source)
        self.assertIn("input-panel-title", source)
        self.assertNotIn("st.sidebar", source)
        self.assertNotIn("control-card", source)

    def test_layout_uses_wider_content_density(self) -> None:
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        css = _build_theme_css()

        self.assertIn("max-width: 1760px", css)
        self.assertIn("padding-left: 1rem", css)
        self.assertIn("padding-right: 1rem", css)
        self.assertIn("st.columns([0.22, 0.78], gap=\"medium\")", source)
        self.assertIn("st.columns([1.1, 0.9], gap=\"medium\")", source)

    def test_theme_css_defines_a2_light_workbench_classes(self) -> None:
        css = _build_theme_css()

        for class_name in ("workbench-note", "stable-zone", "model-pill"):
            with self.subTest(class_name=class_name):
                self.assertIn(class_name, css)

    def test_theme_css_defines_cohesive_palette_tokens(self) -> None:
        css = _build_theme_css()

        expected_tokens = [
            "--primary: #2D6F73",
            "--primary-hover: #245C60",
            "--secondary: #496F8F",
            "--warning: #9B6A2F",
            "--table-header: #EEF4F6",
            "--table-alt: #F7FAFB",
            "--table-hover: #EDF6F6",
        ]
        for token in expected_tokens:
            with self.subTest(token=token):
                self.assertIn(token, css)

    def test_ranked_tables_use_light_html_renderer(self) -> None:
        source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
        renderer = getattr(app, "_format_results_table_html", None)

        self.assertTrue(callable(renderer))
        self.assertNotIn("st.dataframe", source)

        frame = pd.DataFrame(
            [
                {
                    "Mod Temp (C)": 18.0,
                    "Nozzle temp (C)": 25.0,
                    "Predicted Volume (mL)": 1.5179,
                    "Abs Error (mL)": 0.0179,
                }
            ]
        )
        html = renderer(frame, max_height_px=180)

        self.assertIn("results-table-wrap", html)
        self.assertIn("results-table", html)
        self.assertIn("max-height: 180px", html)
        self.assertIn("25.0", html)
        self.assertNotIn("25.0000", html)
        self.assertNotIn("#000", html)

    def test_theme_css_overrides_nested_button_and_select_text(self) -> None:
        css = _build_theme_css()

        expected_selectors = [
            '.stDownloadButton button div[data-testid="stMarkdownContainer"] p',
            '[data-testid="stSelectbox"] [data-baseweb="select"] *',
            '[role="listbox"]',
            '[role="option"]',
            '[role="option"]:not(:hover):not([aria-selected="true"])',
        ]
        for selector in expected_selectors:
            with self.subTest(selector=selector):
                self.assertIn(selector, css)

        self.assertIn("background: var(--panel) !important", css)

    def test_plot_theme_defines_readable_axis_and_legend_text(self) -> None:
        fig = go.Figure()

        _apply_plot_theme(fig)

        self.assertEqual(fig.layout.xaxis.title.font.color, "#24313A")
        self.assertEqual(fig.layout.yaxis.title.font.color, "#24313A")
        self.assertEqual(fig.layout.xaxis.tickfont.color, "#63717B")
        self.assertEqual(fig.layout.yaxis.tickfont.color, "#63717B")
        self.assertEqual(fig.layout.legend.font.color, "#24313A")


if __name__ == "__main__":
    unittest.main()
