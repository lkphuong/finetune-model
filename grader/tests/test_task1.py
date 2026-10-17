import pytest

from config import RUBRIC_MAX
from metrics import grading_metrics
from policy import apply_grading_policy, extract_function_body, is_attempted, prerequisite_function_names
from postprocess import parse_rubric, to_submission_output

FALLBACK = {k: 0 for k in RUBRIC_MAX}
FULL = {"compilable": 1, "io_format": 1, "logic": 4, "edge_case": 2, "complexity": 1, "code_quality": 1}
MULTI_EXAM = {"problems": [
    {"pid": "P1", "prototype": "General* prob1(string filename, int& n);", "is_prerequisite_for": ["P2"]},
    {"pid": "P2", "prototype": "Node* prob2(General* g, int n);"},
]}
SINGLE_EXAM = {}
IMPLEMENTED = "General* prob1(string f, int& n) { n = 0; ifstream in(f); while (in) { n++; } return nullptr; }"
EMPTY = "General* prob1(string f, int& n) { }"
STUB = "General* prob1(string f, int& n) { return NULL; }"


def test_should_clamp_and_sum_when_rubric_values_out_of_range():
    rubric = parse_rubric('{"compilable": 3, "io_format": 1, "logic": -2, "edge_case": 2, "complexity": 1, "code_quality": 1}', FALLBACK)
    assert rubric["compilable"] == 1 and rubric["logic"] == 0
    assert to_submission_output(rubric)["total_score"] == 6


def test_should_use_fallback_when_generation_is_not_json():
    assert parse_rubric("xin chao", {**FALLBACK, "logic": 2})["logic"] == 2


def test_should_use_fallback_when_json_is_not_an_object():
    assert parse_rubric("[1, 2]", FALLBACK) == FALLBACK


def test_should_find_prerequisite_name_when_exam_declares_one():
    assert prerequisite_function_names(MULTI_EXAM) == ["prob1"]


def test_should_extract_nested_body_when_braces_are_nested():
    assert "while" in extract_function_body(IMPLEMENTED, "prob1")


@pytest.mark.parametrize("code,expected", [(IMPLEMENTED, True), (EMPTY, False), (STUB, False), ("int x;", False)])
def test_should_detect_attempt_when_prerequisite_body_varies(code, expected):
    assert is_attempted(code, "prob1") is expected


def test_should_ignore_commented_code_when_checking_attempt():
    assert not is_attempted("/* x */ General* prob1(string f, int& n) { // todo\n }", "prob1")


def test_should_zero_solution_dimensions_when_prerequisite_is_empty():
    graded = apply_grading_policy(FULL, MULTI_EXAM, EMPTY)
    assert graded["logic"] == graded["edge_case"] == 0 and graded["compilable"] == 1


def test_should_keep_rubric_when_prerequisite_is_attempted():
    assert apply_grading_policy(FULL, MULTI_EXAM, IMPLEMENTED) == FULL


def test_should_zero_everything_when_not_compilable():
    assert sum(apply_grading_policy({**FULL, "compilable": 0}, SINGLE_EXAM, "int main(){}").values()) == 0


def test_should_not_apply_prerequisite_rule_when_exam_is_single_problem():
    assert apply_grading_policy(FULL, SINGLE_EXAM, "int main(){}") == FULL


def test_should_not_mutate_input_rubric_when_policy_applies():
    rubric = dict(FULL)
    apply_grading_policy(rubric, MULTI_EXAM, EMPTY)
    assert rubric == FULL


def test_should_report_perfect_metrics_when_prediction_equals_gold():
    gold = [to_submission_output(FULL), to_submission_output({**FULL, "logic": 1})]
    metrics = grading_metrics(gold, gold)
    assert metrics["mae"] == 0 and metrics["exact_logic"] == 1
