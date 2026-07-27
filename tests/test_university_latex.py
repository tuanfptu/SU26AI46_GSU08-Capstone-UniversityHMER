import unittest

from tamer.university.latex import (
    categorize_formula,
    latex_is_syntactically_valid,
    normalize_and_tokenize,
)


class UniversityLatexTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            r"\int", r"\lim", r"\rightarrow", r"\infty", r"\frac", r"\sin",
            "_", "^", "{", "}", "(", ")", "0", "1", "2", "x", "d", "=",
        }

    def test_unbraced_scripts_are_canonicalized(self):
        tokens, reason = normalize_and_tokenize(r"\int_0^1x^2dx", self.vocabulary)
        self.assertIsNone(reason)
        self.assertEqual(
            tokens,
            [r"\int", "_", "{", "0", "}", "^", "{", "1", "}", "x", "^", "{", "2", "}", "d", "x"],
        )

    def test_plain_math_functions_are_normalized(self):
        tokens, reason = normalize_and_tokenize(
            r"lim_{x\to0}\frac{sin(x)}{x}=1", self.vocabulary
        )
        self.assertIsNone(reason)
        self.assertIn(r"\lim", tokens)
        self.assertIn(r"\sin", tokens)
        self.assertIn(r"\rightarrow", tokens)

    def test_oov_is_rejected(self):
        tokens, reason = normalize_and_tokenize(r"\nabla x", self.vocabulary)
        self.assertIsNone(tokens)
        self.assertTrue(reason.startswith("oov:"))

    def test_categories(self):
        self.assertEqual(categorize_formula(r"\int_0^1x dx"), "integral")
        self.assertEqual(categorize_formula(r"lim_{x\to0}sin(x)/x"), "limit")
        self.assertEqual(categorize_formula(r"\frac{\partial f}{\partial x}"), "derivative")

    def test_validity(self):
        self.assertTrue(latex_is_syntactically_valid(["x", "^", "{", "2", "}"]))
        self.assertFalse(latex_is_syntactically_valid(["x", "^", "{", "2"]))


if __name__ == "__main__":
    unittest.main()
