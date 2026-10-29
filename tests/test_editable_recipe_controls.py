import unittest

import numpy as np

from sissimat.pages.spectral_processing import _editable_recipe_control_state
from sissimat.processing.models import Spectrum


class EditableRecipeControlTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(1000.0, 1100.0, 101)
        self.source = Spectrum(x, np.linspace(1.0, 2.0, x.size), name="sample")

    def test_recipe_populates_complete_interface_state(self):
        steps = [
            {
                "operation": "crop",
                "parameters": {"lower": 1010.0, "upper": 1090.0},
            },
            {"operation": "resample", "parameters": {"spacing": 0.5}},
            {
                "operation": "background_normalization",
                "parameters": {"binding": "background"},
            },
            {
                "operation": "transmittance_to_absorbance",
                "parameters": {"percent": True},
            },
            {
                "operation": "savitzky_golay",
                "parameters": {"window": 9, "polynomial_order": 2},
            },
            {"operation": "derivative", "parameters": {"order": 1}},
            {
                "operation": "baseline_correction",
                "parameters": {
                    "method": "Polynomial",
                    "degree": 3,
                    "edge_fraction": 0.2,
                },
            },
            {
                "operation": "normalize",
                "parameters": {
                    "method": "Vector",
                    "lower": 1020.0,
                    "upper": 1080.0,
                },
            },
        ]

        state, warnings = _editable_recipe_control_state(steps, self.source)

        self.assertEqual(warnings, [])
        self.assertTrue(state["lab_enable_crop"])
        self.assertEqual(state["lab_crop_slider"], (1010.0, 1090.0))
        self.assertTrue(state["lab_enable_resample"])
        self.assertEqual(state["lab_resample_spacing"], 0.5)
        self.assertTrue(state["lab_enable_background"])
        self.assertEqual(
            state["lab_representation"],
            "Percentage transmittance → absorbance",
        )
        self.assertTrue(state["lab_enable_smoothing"])
        self.assertEqual(state["lab_smoothing_window"], 9)
        self.assertEqual(state["lab_smoothing_order"], 2)
        self.assertEqual(state["lab_derivative"], "First derivative")
        self.assertEqual(state["lab_baseline_method"], "Polynomial")
        self.assertEqual(state["lab_polynomial_degree"], 3)
        self.assertEqual(state["lab_polynomial_edges"], 0.2)
        self.assertEqual(state["lab_normalization_method"], "Vector")
        self.assertEqual(state["lab_normalization_range"], (1020.0, 1080.0))

    def test_custom_step_order_is_rejected_instead_of_silently_changed(self):
        steps = [
            {"operation": "derivative", "parameters": {"order": 1}},
            {
                "operation": "crop",
                "parameters": {"lower": 1010.0, "upper": 1090.0},
            },
        ]

        with self.assertRaisesRegex(ValueError, "custom step order"):
            _editable_recipe_control_state(steps, self.source)

    def test_recipe_ranges_are_clipped_with_warning(self):
        steps = [
            {
                "operation": "crop",
                "parameters": {"lower": 900.0, "upper": 1200.0},
            }
        ]

        state, warnings = _editable_recipe_control_state(steps, self.source)

        self.assertEqual(state["lab_crop_slider"], (1000.0, 1100.0))
        self.assertEqual(len(warnings), 1)
        self.assertIn("crop range", warnings[0])


if __name__ == "__main__":
    unittest.main()
