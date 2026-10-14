import unittest

from pcnet.config import (
    EXPERIMENT_ORDER,
    EXPERIMENTS,
    REFERENCE_BATCH_SIZE,
    REFERENCE_EPOCHS,
    REFERENCE_IMAGE_SIZE,
    REFERENCE_LEARNING_RATE,
    REFERENCE_SEED,
    REFERENCE_WEIGHT_DECAY,
    selection_epochs,
)


class ProtocolTests(unittest.TestCase):
    def test_four_experiments_are_exact(self):
        self.assertEqual(
            EXPERIMENT_ORDER, ("pcnet", "cph", "rspr", "cph_rspr")
        )
        truth_table = {
            key: (
                spec.challenge_perception,
                spec.rspr,
            )
            for key, spec in EXPERIMENTS.items()
        }
        self.assertEqual(
            truth_table,
            {
                "pcnet": (False, False),
                "cph": (True, False),
                "rspr": (False, True),
                "cph_rspr": (True, True),
            },
        )

    def test_reference_defaults(self):
        self.assertEqual(REFERENCE_EPOCHS, 80)
        self.assertEqual(REFERENCE_BATCH_SIZE, 4)
        self.assertEqual(REFERENCE_IMAGE_SIZE, 384)
        self.assertEqual(REFERENCE_LEARNING_RATE, 1e-5)
        self.assertEqual(REFERENCE_WEIGHT_DECAY, 1e-4)
        self.assertEqual(REFERENCE_SEED, 42)

    def test_test_selection_schedule(self):
        self.assertEqual(selection_epochs(80), (1, *range(60, 81)))
        self.assertEqual(selection_epochs(100), (1, *range(60, 81)))

if __name__ == "__main__":
    unittest.main()
