import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    import torch
    from PIL import Image, ImageDraw

    from pcnet.data import SaliencyTrainDataset

    IMPORT_ERROR = None
except ImportError as error:
    IMPORT_ERROR = error


@unittest.skipIf(IMPORT_ERROR is not None, "Data dependencies unavailable: {}".format(IMPORT_ERROR))
class RSPRBoundaryTests(unittest.TestCase):
    def test_clean_boundary_is_binary_at_training_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb_root, mask_root, thermal_root = (
                root / name for name in ("RGB", "GT", "T")
            )
            for directory in (rgb_root, mask_root, thermal_root):
                directory.mkdir()
            Image.new("RGB", (16, 16), "gray").save(rgb_root / "sample.png")
            Image.new("RGB", (16, 16), "gray").save(thermal_root / "sample.png")
            clean_mask = Image.new("L", (16, 16), 0)
            ImageDraw.Draw(clean_mask).rectangle((4, 4, 11, 11), fill=255)
            clean_mask.save(mask_root / "sample.png")

            dataset = SaliencyTrainDataset(
                rgb_root, mask_root, thermal_root, image_size=8,
                include_boundary=True,
            )
            with (
                patch("pcnet.data.random_flip", side_effect=lambda *items: items),
                patch("pcnet.data.random_crop", side_effect=lambda *items: items),
                patch("pcnet.data.random_rotation", side_effect=lambda *items: items),
                patch("pcnet.data.color_enhance", side_effect=lambda image: image),
                patch(
                    "pcnet.data.salt_and_pepper",
                    side_effect=lambda mask: Image.new("L", mask.size, 0),
                ),
            ):
                sample = dataset[0]

            boundary = sample["boundary"]
            self.assertEqual(tuple(boundary.shape), (1, 8, 8))
            self.assertTrue(torch.all((boundary == 0) | (boundary == 1)))
            self.assertGreater(int(boundary.sum().item()), 0)
            self.assertEqual(int(sample["mask"].sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
