import unittest

import numpy as np

from sissimat.processing.io import load_spectra, preferred_dataset_label


class TextLoadingTests(unittest.TestCase):
    def test_prefers_ssc_dataset_over_interferogram(self):
        labels = ["SIFG", "SSC", "AB"]

        self.assertEqual(preferred_dataset_label(labels), "SSC")

    def test_loads_comma_separated_data_with_header(self):
        contents = b"wavenumber,intensity\n1000,0.1\n1001,0.2\n1002,0.3\n"

        spectra = load_spectra(contents, "example.csv")

        spectrum = spectra["Data"]
        np.testing.assert_allclose(spectrum.x, [1000.0, 1001.0, 1002.0])
        np.testing.assert_allclose(spectrum.y, [0.1, 0.2, 0.3])

    def test_loads_tab_separated_data(self):
        contents = b"1000\t1\n1001\t2\n1002\t3\n"

        spectra = load_spectra(contents, "example.txt")

        np.testing.assert_allclose(spectra["Data"].y, [1.0, 2.0, 3.0])

    def test_rejects_non_numeric_text(self):
        with self.assertRaises(ValueError):
            load_spectra(b"this is not a spectrum", "invalid.txt")


if __name__ == "__main__":
    unittest.main()
