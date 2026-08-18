import unittest

import numpy as np

from sissimat.processing.models import Spectrum
from sissimat.processing.recipe import apply_recipe, required_bindings, validate_recipe_document


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(1000.0, 1100.0, 101)
        self.sample = Spectrum(self.x, np.full(self.x.size, 5.0), name="sample")
        self.background = Spectrum(self.x, np.full(self.x.size, 10.0), name="background")

    def test_applies_background_absorbance_crop_and_normalization(self):
        recipe = {
            "schema_version": 1,
            "steps": [
                {
                    "operation": "background_normalization",
                    "parameters": {"binding": "background"},
                },
                {
                    "operation": "transmittance_to_absorbance",
                    "parameters": {"percent": False},
                },
                {
                    "operation": "crop",
                    "parameters": {"lower": 1020.0, "upper": 1080.0},
                },
            ],
        }

        result = apply_recipe(self.sample, recipe, bindings={"background": self.background})

        np.testing.assert_allclose(result.spectrum.y, -np.log10(0.5))
        self.assertGreaterEqual(result.spectrum.x.min(), 1020.0)
        self.assertLessEqual(result.spectrum.x.max(), 1080.0)

    def test_reports_required_background_binding(self):
        recipe = {
            "schema_version": 1,
            "steps": [
                {
                    "operation": "background_normalization",
                    "parameters": {"binding": "experiment_background"},
                }
            ],
        }

        self.assertEqual(required_bindings(recipe), {"experiment_background"})

    def test_recipe_converts_transmittance_above_unity_and_reports_warning(self):
        sample = Spectrum(self.x, np.linspace(9.0, 10.24, self.x.size), name="sample")
        recipe = {
            "schema_version": 1,
            "steps": [
                {"operation": "background_normalization", "parameters": {}},
                {"operation": "transmittance_to_absorbance", "parameters": {"percent": False}},
            ],
        }

        result = apply_recipe(sample, recipe, bindings={"background": self.background})

        self.assertLess(float(result.spectrum.y.min()), 0.0)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("maximum T = 1.024", result.warnings[0])

    def test_missing_binding_has_step_context(self):
        recipe = {
            "schema_version": 1,
            "steps": [{"operation": "background_normalization", "parameters": {}}],
        }

        with self.assertRaisesRegex(ValueError, "Recipe step 1"):
            apply_recipe(self.sample, recipe)

    def test_rejects_unknown_operations(self):
        with self.assertRaisesRegex(ValueError, "unsupported operation"):
            validate_recipe_document(
                {"schema_version": 1, "steps": [{"operation": "atmospheric_compensation"}]}
            )


if __name__ == "__main__":
    unittest.main()
