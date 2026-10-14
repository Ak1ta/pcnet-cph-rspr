import tempfile
import unittest
from pathlib import Path


try:
    import numpy as np
    from PIL import Image

    from pcnet.evaluation import (
        METRIC_NAMES,
        evaluate_directory,
        released_pcnet_emeasure,
    )

    BASIC_IMPORT_ERROR = None
except ImportError as error:
    BASIC_IMPORT_ERROR = error

try:
    import py_sod_metrics  # noqa: F401

    METRIC_IMPORT_ERROR = BASIC_IMPORT_ERROR
except ImportError as error:
    METRIC_IMPORT_ERROR = error


@unittest.skipIf(
    METRIC_IMPORT_ERROR is not None,
    "Metric dependencies unavailable: {}".format(METRIC_IMPORT_ERROR),
)
class MetricTests(unittest.TestCase):
    def test_perfect_prediction_returns_five_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_root = root / "pred"
            mask_root = root / "gt"
            prediction_root.mkdir()
            mask_root.mkdir()
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            Image.fromarray(mask).save(mask_root / "sample.png")
            Image.fromarray(mask).save(prediction_root / "sample.png")
            metrics = evaluate_directory(prediction_root, mask_root)
            self.assertEqual(tuple(metrics), METRIC_NAMES)
            self.assertEqual(metrics["mae"], 0.0)


@unittest.skipIf(
    BASIC_IMPORT_ERROR is not None,
    "Array/image dependencies unavailable: {}".format(BASIC_IMPORT_ERROR),
)
class EmeasureFormulaTests(unittest.TestCase):
    def test_released_formula_preserves_n_minus_one_denominator(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, 2:] = 255
        self.assertAlmostEqual(
            released_pcnet_emeasure(mask, mask), 16.0 / 15.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
