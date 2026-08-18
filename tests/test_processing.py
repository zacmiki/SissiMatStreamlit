import unittest

import numpy as np

from sissimat.processing.models import Spectrum
from sissimat.processing.transforms import (
    baseline_correct,
    background_normalize,
    crop,
    derivative,
    normalize,
    resample,
    savgol_smooth,
    to_absorbance,
    transmittance_warning,
)


class SpectrumModelTests(unittest.TestCase):
    def test_cleaned_sorts_and_averages_duplicate_coordinates(self):
        spectrum = Spectrum(
            x=np.array([3.0, 2.0, 2.0, 1.0]),
            y=np.array([30.0, 18.0, 22.0, 10.0]),
        )

        cleaned = spectrum.cleaned()

        np.testing.assert_allclose(cleaned.x, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(cleaned.y, [10.0, 20.0, 30.0])


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(400.0, 4000.0, 401)
        self.spectrum = Spectrum(self.x, 2.0 + 0.001 * self.x, name="synthetic")

    def test_percentage_transmittance_to_absorbance(self):
        transmittance = Spectrum(self.x, np.full(self.x.size, 10.0))

        converted = to_absorbance(transmittance, percent=True)

        np.testing.assert_allclose(converted.y, 1.0)
        self.assertEqual(converted.y_label, "Absorbance")

    def test_absorbance_rejects_nonpositive_transmittance(self):
        transmittance = Spectrum(self.x, np.linspace(0.0, 1.0, self.x.size))

        with self.assertRaises(ValueError):
            to_absorbance(transmittance)

    def test_absorbance_allows_values_above_unity_with_warning(self):
        transmittance = Spectrum(self.x, np.linspace(0.8, 1.024, self.x.size))

        converted = to_absorbance(transmittance)
        warning = transmittance_warning(transmittance)

        np.testing.assert_allclose(converted.y, -np.log10(transmittance.y))
        self.assertLess(float(converted.y.min()), 0.0)
        self.assertIn("maximum T = 1.024", warning)

    def test_crop_and_resample(self):
        cropped = crop(self.spectrum, 1000.0, 2000.0)
        sampled = resample(cropped, 10.0)

        self.assertGreaterEqual(sampled.x[0], 1000.0)
        self.assertLessEqual(sampled.x[-1], 2000.0)
        np.testing.assert_allclose(np.diff(sampled.x), 10.0)

    def test_savgol_smoothing_preserves_linear_signal(self):
        smoothed = savgol_smooth(self.spectrum, window=11, order=2)

        np.testing.assert_allclose(smoothed.y, self.spectrum.y, atol=1e-10)

    def test_first_and_second_derivatives(self):
        quadratic = Spectrum(self.x, self.x**2)

        first = derivative(quadratic, 1)
        second = derivative(quadratic, 2)

        np.testing.assert_allclose(first.y, 2 * self.x, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(second.y, 2.0, rtol=1e-10, atol=1e-9)

    def test_normalization_methods(self):
        minmax = normalize(self.spectrum, "min-max")
        vector = normalize(self.spectrum, "vector")
        area = normalize(self.spectrum, "area")

        self.assertAlmostEqual(float(minmax.y.min()), 0.0)
        self.assertAlmostEqual(float(minmax.y.max()), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(vector.y)), 1.0)
        self.assertAlmostEqual(float(np.trapz(area.y, area.x)), 1.0)

    def test_rubberband_removes_linear_baseline_under_positive_peak(self):
        x = np.linspace(-1.0, 1.0, 501)
        baseline = 2.0 + 0.5 * x
        peak = np.exp(-(x**2) / 0.05)
        spectrum = Spectrum(x, baseline + peak)

        corrected, estimated = baseline_correct(spectrum, "rubber-band")

        np.testing.assert_allclose(estimated, baseline, atol=1e-7)
        self.assertGreater(corrected.y.max(), 0.99)

    def test_background_normalization_interpolates_and_calculates_ratio(self):
        primary = Spectrum(self.x, 4.0 + 2.0 * np.sin(self.x / 100.0))
        background_x = np.linspace(300.0, 4100.0, 801)
        background = Spectrum(background_x, np.full(background_x.size, 2.0))

        result = background_normalize(primary, background)

        np.testing.assert_allclose(result.y, 2.0 + np.sin(self.x / 100.0), atol=1e-12)
        self.assertIn("I / I₀", result.y_label)

    def test_background_normalization_rejects_zero_background(self):
        background = Spectrum(self.x, np.zeros(self.x.size))

        with self.assertRaises(ValueError):
            background_normalize(self.spectrum, background)


if __name__ == "__main__":
    unittest.main()
