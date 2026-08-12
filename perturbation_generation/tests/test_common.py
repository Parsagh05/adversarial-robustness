import os
import unittest


os.environ.setdefault("MVTEC_ROOT", ".")
os.environ.setdefault("VISA_ROOT", ".")
os.environ.setdefault("OUTPUT_BASE", ".")

from common import parse_numeric


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


if __name__ == "__main__":
    unittest.main()
