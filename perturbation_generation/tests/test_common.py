import os
import unittest


os.environ.setdefault("MVTEC_ROOT", ".")
os.environ.setdefault("VISA_ROOT", ".")
os.environ.setdefault("OUTPUT_BASE", ".")

from common import generation_datasets, parse_numeric


class NumericConfigurationTests(unittest.TestCase):
    def test_parses_integer_and_decimal_fraction_expressions(self) -> None:
        self.assertAlmostEqual(parse_numeric("8/255"), 8.0 / 255.0)
        self.assertAlmostEqual(parse_numeric("0.25/255"), 0.25 / 255.0)
        self.assertAlmostEqual(parse_numeric("0.5"), 0.5)

    def test_rejects_invalid_or_zero_denominator(self) -> None:
        with self.assertRaises(ValueError):
            parse_numeric("1/2/3")
        with self.assertRaises(ValueError):
            parse_numeric("1/0")


class DatasetSelectionTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("GENERATION_DATASETS", None)

    def test_accepts_single_or_both_datasets(self) -> None:
        os.environ["GENERATION_DATASETS"] = "mvtec"
        self.assertEqual(generation_datasets(), ("mvtec",))
        os.environ["GENERATION_DATASETS"] = "mvtec,visa"
        self.assertEqual(generation_datasets(), ("mvtec", "visa"))

    def test_rejects_unknown_or_duplicate_datasets(self) -> None:
        for value in ("mvtec,mvtec", "unknown", ""):
            os.environ["GENERATION_DATASETS"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation_datasets()


if __name__ == "__main__":
    unittest.main()
