from __future__ import annotations

import unittest

from sampler_optimizer import build_feature_frame


class SamplerOptimizerTest(unittest.TestCase):
    def test_build_feature_frame_computes_particles_squared(self) -> None:
        frame = build_feature_frame(
            rh_percent=38.5,
            ambient_temp_c=23.5,
            particles=2500.0,
            mod_temp_c=18.0,
            nozzle_temp_c=22.0,
            collection_medium=0,
            feature_columns=["Particles", "Particles_squared"],
        )

        self.assertEqual(frame.loc[0, "Particles"], 2500.0)
        self.assertEqual(frame.loc[0, "Particles_squared"], 2500.0**2)

    def test_build_feature_frame_rejects_missing_engineered_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing engineered feature columns: NotAFeature"):
            build_feature_frame(
                rh_percent=38.5,
                ambient_temp_c=23.5,
                particles=2500.0,
                mod_temp_c=18.0,
                nozzle_temp_c=22.0,
                collection_medium=0,
                feature_columns=["Particles", "NotAFeature"],
            )


if __name__ == "__main__":
    unittest.main()
