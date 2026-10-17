"""Deterministic enforcement of exam grading policy on top of model output.

Rules (both come from the exam rules, not from learned weights):
1. A program that does not compile earns nothing.
2. In a multi-problem exam, if a prerequisite problem was not attempted, no later problem is graded.
"""
import re

from config import RUBRIC_MAX, SOLUTION_DIMENSIONS

LINE_COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
FUNCTION_NAME = re.compile(r"(\w+)\s*\(")
STUB_BODY = re.compile(r"^(return\b[^;]*;)?$")


def strip_comments(code: str) -> str:
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub("", code))


def extract_function_body(code: str, name: str) -> str | None:
    """Return the text between the braces of the first definition of `name`, or None if absent."""
    # A definition is `name(...) {` while a call or declaration ends with `;` or `)` before no brace
    for match in re.finditer(rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", code):
        depth, pos = 1, match.end()
        while pos < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[pos], 0)
            pos += 1
        return code[match.end() : pos - 1]
    return None


def is_attempted(code: str, function_name: str) -> bool:
    body = extract_function_body(strip_comments(code), function_name)
    return body is not None and not STUB_BODY.match(re.sub(r"\s+", " ", body).strip())


def prerequisite_function_names(exam: dict) -> list[str]:
    problems = exam.get("problems") or []
    return [
        FUNCTION_NAME.search(p["prototype"]).group(1)
        for p in problems
        if p.get("is_prerequisite_for")
    ]


def zero_solution_dimensions(rubric: dict) -> dict:
    return {**rubric, **{name: 0 for name in SOLUTION_DIMENSIONS}}


def apply_grading_policy(rubric: dict, exam: dict, code: str) -> dict:
    """Return a rubric consistent with the exam policy; the input rubric is not mutated."""
    if rubric["compilable"] == 0:
        return {name: 0 for name in RUBRIC_MAX}
    if all(is_attempted(code, fn) for fn in prerequisite_function_names(exam)):
        return rubric
    return zero_solution_dimensions(rubric)
