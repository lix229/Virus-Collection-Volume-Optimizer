from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


class StandaloneRuntimeTest(unittest.TestCase):
    def test_optimizer_runs_from_copied_directory_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir) / "streamlit_app"
            shutil.copytree(
                APP_ROOT,
                temp_root,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

            script = textwrap.dedent(
                """
                from sampler_optimizer import load_model_artifacts, optimize_temperature_grid

                model, feature_columns, _ = load_model_artifacts()
                results = optimize_temperature_grid(
                    model=model,
                    feature_columns=feature_columns,
                    rh_percent=38.5,
                    ambient_temp_c=23.5,
                    particles=2500.0,
                    collection_medium=0,
                    mod_temp_values=(18.0, 19.0),
                    nozzle_temp_values=(22.0, 23.0),
                    target_volume_ml=1.5,
                )
                assert not results.empty
                print(results.head(1).to_dict(orient="records")[0]["Predicted Volume (mL)"])
                """
            )

            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=temp_root,
                capture_output=True,
                text=True,
            )

            if completed.returncode != 0:
                self.fail(
                    "Standalone runtime failed.\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )


if __name__ == "__main__":
    unittest.main()
