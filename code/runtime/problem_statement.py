"""problem_statement.py

Turn rows of an Abalone CSV into the English problem statement written to the problem_spec section.

Responsibilities:
- Read a CSV with the dataset's columns and decide the task of each row from the column that is missing: no `sex`
  -> task "sex"; no `number_of_rings` -> task "age". When a row has both, the caller's task decides which column is
  hidden; the hidden value is kept as the truth for evaluations and never reaches the agents.
- Validate the values (positive, in dataset units, whole ring counts, sex in M/F/I) with actionable messages.
- Render templates/abalone_{task}_problem.md with two blocks of values, each addressed to its readers: the dataset
  values exactly as in the CSV (what the tools were trained on) for the tool agents, and the same values multiplied by
  200 (millimetres and grams, the units of the articles) for the article experts. The template does the conversion,
  so no model has to.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
SCALE = 200
MEASUREMENTS = ("length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight")
LENGTHS = ("length", "diameter", "height")
RINGS = "number_of_rings"
SEX = "sex"
SEX_VALUES = ("M", "F", "I")
SEX_WORDS = {"M": "male", "F": "female", "I": "infant (immature)"}
RINGS_TO_YEARS = 1.5

VARIABLES_TEXT = (
    "Length is the longest shell measurement, and diameter is measured perpendicular\n"
    "to it. Height is measured with the meat in the shell. Whole weight is the weight\n"
    "of the whole abalone, shucked weight the weight of the meat, viscera weight the\n"
    "gut weight after bleeding, and shell weight the weight of the shell after\n"
    "drying. The number of rings counts the growth rings in the shell; adding 1.5\n"
    "gives the age in years."
)
ABOUT_TEXT = (
    "Abalone are marine snails (gastropod molluscs). These records come from a\n"
    "fisheries study of blacklip abalone (Haliotis rubra) on the north coast and\n"
    "islands of Bass Strait, Tasmania, Australia (Nash et al. 1994). Their age is\n"
    "found by cutting the shell through the cone, staining it and counting the rings\n"
    "under a microscope, a slow task, so easier measurements are used to predict it.\n"
    "Sex is M (male), F (female) or I (infant: immature, with no testis or ovary\n"
    "visible yet)."
)


class RowError(ValueError):
    """A CSV row cannot become a problem; the message names the row and the fix."""


@dataclass
class Problem:
    row_number: int  # 0-based position among the CSV's data rows
    task: str
    values: dict[str, str]  # the CSV strings, exactly as written, for the columns the agents may see
    truth: dict[str, Any] = field(default_factory=dict)
    statement: str = ""

    def specimen(self) -> dict[str, Any]:
        """Typed values in dataset units, as the tools expect them."""
        typed: dict[str, Any] = {name: float(self.values[name]) for name in MEASUREMENTS}
        if self.task == "sex":
            typed[RINGS] = int(float(self.values[RINGS]))
        else:
            typed[SEX] = self.values[SEX]
        return typed


def infer_task(columns: set[str], requested: str | None) -> str:
    missing = [name for name in MEASUREMENTS if name not in columns]
    if missing:
        raise RowError(f"The CSV lacks the measurement column(s) {missing}; every row needs {list(MEASUREMENTS)}.")
    has_sex, has_rings = SEX in columns, RINGS in columns
    if has_sex and has_rings:
        if requested not in ("sex", "age"):
            raise RowError("The CSV has both 'sex' and 'number_of_rings'; pass --task sex or --task age to choose which one to hide.")
        return requested
    if has_rings:
        task = "sex"
    elif has_sex:
        task = "age"
    else:
        raise RowError("The CSV needs 'number_of_rings' (to estimate the sex) or 'sex' (to estimate the age).")
    if requested and requested != task:
        raise RowError(f"--task {requested} needs the column {'sex' if requested == 'age' else 'number_of_rings'}, which this CSV lacks.")
    return task


def check_value(name: str, raw: str, row_number: int) -> None:
    where = f"row {row_number}, column {name}"
    if name == SEX:
        if raw not in SEX_VALUES:
            raise RowError(f"{where}: sex must be one of {list(SEX_VALUES)}, got {raw!r}.")
        return
    try:
        value = float(raw)
    except ValueError:
        raise RowError(f"{where}: {raw!r} is not a number.") from None
    if not math.isfinite(value) or value <= 0:
        raise RowError(f"{where}: {raw!r} must be a positive number.")
    if name == RINGS:
        if not value.is_integer():
            raise RowError(f"{where}: the ring count must be a whole number, got {raw!r}.")
    elif value > 5:
        raise RowError(f"{where}: {raw!r} looks like millimetres or grams; the CSV must hold dataset units (divided by 200).")


def original_unit(name: str, raw: str) -> str:
    value = round(float(raw) * SCALE, 2)
    unit = "mm" if name in LENGTHS else "g"
    return f"{value:g} {unit}"


def render(problem: Problem) -> str:
    dataset_lines = [f"- {name}: {problem.values[name]}" for name in MEASUREMENTS]
    original_lines = [f"- {name.replace('_', ' ')}: {original_unit(name, problem.values[name])}" for name in MEASUREMENTS]
    if problem.task == "sex":
        rings = int(float(problem.values[RINGS]))
        dataset_lines.append(f"- number_of_rings: {problem.values[RINGS]}")
        original_lines.append(f"- growth rings: {rings} (the dataset takes age as rings + 1.5, about {rings + RINGS_TO_YEARS:g} years)")
        template = (TEMPLATES_DIR / "abalone_sex_problem.md").read_text(encoding="utf-8")
        sex = ""
    else:
        sex = problem.values[SEX]
        dataset_lines.append(f"- sex: {sex}")
        original_lines.append(f"- sex: {SEX_WORDS[sex]} ({sex})")
        template = (TEMPLATES_DIR / "abalone_age_problem.md").read_text(encoding="utf-8")
    replacements = {
        "{{dataset_values}}": "\n".join(dataset_lines),
        "{{original_values}}": "\n".join(original_lines),
        "{{variables_text}}": VARIABLES_TEXT,
        "{{about_text}}": ABOUT_TEXT,
        "{{sex}}": sex,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template.strip() + "\n"


def problems_from_csv(path: Path, task: str | None = None, rows: list[int] | None = None) -> list[Problem]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = {name.strip() for name in reader.fieldnames or []}
        chosen_task = infer_task(columns, task)
        records = list(reader)
    wanted = range(len(records)) if rows is None else rows
    problems = []
    for row_number in wanted:
        if not 0 <= row_number < len(records):
            raise RowError(f"Row {row_number} does not exist; the CSV has {len(records)} data rows (0-based).")
        record = {key.strip(): (value or "").strip() for key, value in records[row_number].items()}
        hidden = SEX if chosen_task == "sex" else RINGS
        visible = [*MEASUREMENTS, RINGS if chosen_task == "sex" else SEX]
        for name in visible:
            check_value(name, record.get(name, ""), row_number)
        truth = {}
        if record.get(hidden):
            truth[hidden] = record[hidden] if hidden == SEX else int(float(record[hidden]))
        problem = Problem(row_number, chosen_task, {name: record[name] for name in visible}, truth)
        problem.statement = render(problem)
        problems.append(problem)
    return problems
