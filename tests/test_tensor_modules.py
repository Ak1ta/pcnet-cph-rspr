import unittest


try:
    import torch

    from pcnet.config import ModelConfig
    from pcnet.model import RegionStructurePrototypeRectifier

    IMPORT_ERROR = None
except ImportError as error:
    IMPORT_ERROR = error


@unittest.skipIf(IMPORT_ERROR is not None, "Tensor dependencies unavailable: {}".format(IMPORT_ERROR))
class TensorModuleTests(unittest.TestCase):
    def test_rspr_starts_as_exact_residual_identity(self):
        config = ModelConfig(experiment="rspr", rspr_channels=16)
        module = RegionStructurePrototypeRectifier(config).eval()
        decoder = torch.randn(2, 64, 8, 8)
        rgb = torch.randn(2, 128, 8, 8)
        coarse = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            output = module(decoder, rgb, coarse)
        self.assertTrue(torch.equal(output["saliency"], coarse))
        self.assertTrue(torch.count_nonzero(output["correction"]) == 0)


if __name__ == "__main__":
    unittest.main()
