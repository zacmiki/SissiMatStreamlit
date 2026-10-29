import json
import unittest

from streamlit.testing.v1 import AppTest


class EditableRecipeUITests(unittest.TestCase):
    def test_imported_recipe_prefills_controls_and_remains_editable(self):
        source = "wavenumber,intensity\n" + "\n".join(
            f"{value},{1.0 + value / 1000.0}" for value in range(1000, 1101)
        )
        recipe = {
            "schema_version": 1,
            "steps": [
                {
                    "operation": "crop",
                    "parameters": {"lower": 1010.0, "upper": 1090.0},
                },
                {
                    "operation": "savitzky_golay",
                    "parameters": {"window": 9, "polynomial_order": 2},
                },
            ],
        }
        app = AppTest.from_string(
            "from sissimat.pages.spectral_processing import "
            "render_spectral_processing_lab\n"
            "render_spectral_processing_lab()",
            default_timeout=30,
        ).run()

        app.file_uploader[0].upload(
            "sample.csv", source.encode("utf-8"), "text/csv"
        ).run()
        recipe_uploader = app.get_by_key("single_imported_recipe")
        recipe_uploader.upload(
            "starting_recipe.json",
            json.dumps(recipe).encode("utf-8"),
            "application/json",
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(app.get_by_key("lab_enable_crop").value)
        self.assertEqual(app.get_by_key("lab_crop_lower").value, 1010.0)
        self.assertEqual(app.get_by_key("lab_crop_upper").value, 1090.0)
        self.assertTrue(app.get_by_key("lab_enable_smoothing").value)
        self.assertEqual(app.get_by_key("lab_smoothing_window").value, 9)
        self.assertEqual(app.get_by_key("lab_smoothing_order").value, 2)
        processing_sections = [
            section
            for section in app.expander
            if section.label.startswith(("1 ·", "2 ·", "3 ·", "4 ·", "5 ·"))
        ]
        self.assertEqual(len(processing_sections), 5)
        self.assertTrue(all(section.proto.expanded for section in processing_sections))

        app.get_by_key("lab_crop_lower").set_value(1020.0).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.get_by_key("lab_crop_lower").value, 1020.0)
        self.assertTrue(
            any("status: modified" in message.value for message in app.info)
        )


if __name__ == "__main__":
    unittest.main()
