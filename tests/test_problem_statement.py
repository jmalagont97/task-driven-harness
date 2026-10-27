"""test_problem_statement.py

Tests for code/runtime/problem_statement.py: task inference from the CSV columns, validation, and the two blocks of
values (dataset units for the tool agents, original units for the article experts).

    .venv/bin/python -m unittest tests.test_problem_statement -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from runtime.problem_statement import RowError, infer_task, problems_from_csv  # noqa: E402

DATASET = REPO_ROOT / "artifacts" / "datasets" / "abalone_dataset" / "train" / "abalone_dataset.csv"
SMOKE_SEX = REPO_ROOT / "evals" / "abalone_debate" / "smoke_sex.csv"
SMOKE_AGE = REPO_ROOT / "evals" / "abalone_debate" / "smoke_age.csv"
MEASUREMENTS = {"length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"}


def write_csv(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return Path(handle.name)


class TestTaskInference(unittest.TestCase):
    def test_missing_column_decides_the_task(self):
        self.assertEqual(infer_task(MEASUREMENTS | {"number_of_rings"}, None), "sex")
        self.assertEqual(infer_task(MEASUREMENTS | {"sex"}, None), "age")
        self.assertEqual(infer_task(MEASUREMENTS | {"sex", "number_of_rings"}, "age"), "age")

    def test_ambiguous_or_incomplete_columns_are_rejected(self):
        with self.assertRaisesRegex(RowError, "--task"):
            infer_task(MEASUREMENTS | {"sex", "number_of_rings"}, None)
        with self.assertRaisesRegex(RowError, "number_of_rings"):
            infer_task(MEASUREMENTS, None)
        with self.assertRaisesRegex(RowError, "shell_weight"):
            infer_task(MEASUREMENTS - {"shell_weight"} | {"sex"}, None)
        with self.assertRaisesRegex(RowError, "lacks"):
            infer_task(MEASUREMENTS | {"sex"}, "sex")


class TestStatement(unittest.TestCase):
    def test_sex_statement_separates_the_two_audiences(self):
        problem = problems_from_csv(SMOKE_SEX)[0]
        text = problem.statement
        tools_block = text.split("## Measurements for the tool agents (dataset units)")[1].split("## Measurements for the literature experts")[0]
        experts_block = text.split("## Measurements for the literature experts (original units)")[1].split("## What the variables mean")[0]
        for line in ("- length: 0.455", "- shucked_weight: 0.2245", "- number_of_rings: 15"):
            self.assertIn(line, tools_block)
        self.assertNotIn("mm", tools_block.split("\n\n", 2)[-1])
        for line in ("- length: 91 mm", "- diameter: 73 mm", "- whole weight: 102.8 g", "- shucked weight: 44.9 g", "- shell weight: 30 g", "about 16.5 years"):
            self.assertIn(line, experts_block)
        self.assertNotIn("|", text, "the variables are explained in text, not in a table")
        self.assertIn("Answer M, F or I", text)
        self.assertEqual(problem.specimen()["number_of_rings"], 15)

    def test_age_statement_never_shows_the_ring_count(self):
        problem = problems_from_csv(SMOKE_AGE)[0]
        self.assertEqual(problem.task, "age")
        self.assertIn("of sex M", problem.statement)
        self.assertIn("- sex: M", problem.statement)
        self.assertIn("- sex: male (M)", problem.statement)
        self.assertNotIn("number_of_rings:", problem.statement)
        self.assertNotIn("growth rings:", problem.statement)
        self.assertEqual(problem.specimen()["sex"], "M")

    def test_dataset_rows_hide_the_chosen_target_and_keep_it_as_truth(self):
        sex = problems_from_csv(DATASET, "sex", [0])[0]
        self.assertEqual(sex.truth, {"sex": "M"})
        self.assertNotIn("- sex:", sex.statement)
        age = problems_from_csv(DATASET, "age", [0, 1])
        self.assertEqual([p.truth for p in age], [{"number_of_rings": 15}, {"number_of_rings": 7}])
        self.assertTrue(all("number_of_rings:" not in p.statement for p in age))

    def test_bad_values_name_the_row_and_the_fix(self):
        header = "length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight,number_of_rings\n"
        cases = {
            "91,0.365,0.095,0.514,0.2245,0.101,0.15,15\n": "divided by 200",
            "0.455,0.365,0,0.514,0.2245,0.101,0.15,15\n": "positive",
            "0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15.5\n": "whole number",
            "0.455,abc,0.095,0.514,0.2245,0.101,0.15,15\n": "not a number",
        }
        for row, fragment in cases.items():
            with self.subTest(row=row):
                path = write_csv(header + row)
                with self.assertRaisesRegex(RowError, fragment):
                    problems_from_csv(path)
                path.unlink()
        with self.assertRaisesRegex(RowError, "does not exist"):
            problems_from_csv(SMOKE_SEX, rows=[3])


if __name__ == "__main__":
    unittest.main()
