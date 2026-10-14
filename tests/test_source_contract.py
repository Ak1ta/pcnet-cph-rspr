import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_four_explicit_training_entries(self):
        expected = {
            "train_pcnet.py": "pcnet",
            "train_cph.py": "cph",
            "train_rspr.py": "rspr",
            "train_cph_rspr.py": "cph_rspr",
        }
        for filename, experiment in expected.items():
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn('main_for("{}")'.format(experiment), source)

    def test_new_code_has_no_legacy_package_dependency(self):
        forbidden = ("Code.", "CA_PCNet", "CA_RSPR_PCNet")
        paths = list((ROOT / "pcnet").rglob("*.py"))
        paths += [
            ROOT / name
            for name in (
                "train_pcnet.py",
                "train_cph.py",
                "train_rspr.py",
                "train_cph_rspr.py",
                "infer.py",
                "benchmark.py",
                "verify_environment.py",
            )
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, source, str(path))

    def test_metric_contract_is_exact_and_ordered(self):
        tree = ast.parse((ROOT / "pcnet" / "evaluation.py").read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "METRIC_NAMES"
        )
        values = tuple(ast.literal_eval(assignment.value))
        self.assertEqual(values, ("em", "sm", "adpfm", "wfm", "mae"))

    def test_cscr_entry_points_are_removed(self):
        self.assertFalse((ROOT / "train_ca.py").exists())
        self.assertFalse((ROOT / "train_ca_rspr.py").exists())

    def test_output_checkpoint_names_are_unified(self):
        source = (ROOT / "pcnet" / "training.py").read_text(encoding="utf-8")
        for filename in ("best.pth", "final.pth"):
            self.assertIn('"{}"'.format(filename), source)
        self.assertNotIn("last.pth", source)
        self.assertNotIn('"--resume"', source)

    def test_unused_attention_weights_are_not_materialized(self):
        source = (ROOT / "pcnet" / "model.py").read_text(encoding="utf-8")
        self.assertIn("need_weights=False", source)


if __name__ == "__main__":
    unittest.main()
