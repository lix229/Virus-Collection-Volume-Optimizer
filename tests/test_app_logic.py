from __future__ import annotations

import unittest

import pandas as pd

from app import _build_theme_css, _build_temperature_values, _filter_nozzle_ge_mod_results


class AppLogicTest(unittest.TestCase):
    def test_theme_css_defines_dark_inputs_with_visible_text(self) -> None:
        css = _build_theme_css()

        self.assertIn('div[data-testid="stNumberInput"] input', css)
        self.assertIn('color: #e8f1f8', css)
        self.assertIn('background: #213242', css)
        self.assertIn('border: 1px solid #4d6a80', css)
        self.assertIn('box-shadow: 0 0 0 1px #8fd3ff', css)

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


if __name__ == "__main__":
    unittest.main()
