"""Parse model generations into a schema-conformant rubric."""
import json

from config import RUBRIC_MAX


def extract_json(text: str):
    """Return the first JSON object in the text, or None when nothing parses."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_rubric(text: str, fallback: dict) -> dict:
    """Clamp each dimension to its valid range; missing dimensions take the fallback value."""
    raw = extract_json(text)
    raw = raw if isinstance(raw, dict) else {}
    rubric = {}
    for name, upper in RUBRIC_MAX.items():
        value = raw.get(name)
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        rubric[name] = min(max(int(round(value)), 0), upper) if valid else fallback[name]
    return rubric


def to_submission_output(rubric: dict) -> dict:
    return {"total_score": sum(rubric.values()), "rubric": rubric}
