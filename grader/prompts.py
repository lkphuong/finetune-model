"""Prompt and target construction shared by training and inference so both see identical formats."""
import json

from config import (
    INITIAL_CODE_CHARS,
    MAX_COMPILE_LOG_CHARS,
    MAX_FAILED_TESTS_SHOWN,
    MAX_STATEMENT_CHARS,
    MIN_CODE_CHARS,
    RUBRIC_MAX,
)

SYSTEM_PROMPT = (
    "Bạn là trợ giảng chấm bài lập trình C++ cho sinh viên. "
    "Chỉ trả lời bằng đúng định dạng JSON được yêu cầu, không giải thích thêm."
)

INSTRUCTION = (
    "Chấm điểm theo rubric. Trả về JSON với các khóa: "
    + ", ".join(f"{k} (0-{v})" for k, v in RUBRIC_MAX.items())
    + "."
)


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n// ... đã lược bớt ...\n" + text[-half:]


def summarize_test_report(report) -> str:
    if not report:
        return "Không có báo cáo test."
    failed = [t for t in report if not t.get("passed")]
    lines = [f"Đạt {len(report) - len(failed)}/{len(report)} test công khai."]
    for test in failed[:MAX_FAILED_TESTS_SHOWN]:
        lines.append(f"- input={test['input']!r} expected={test['expected']!r} actual={test['actual']!r}")
    return "\n".join(lines)


def describe_grading_policy(exam: dict) -> str:
    """Render the per-problem weights and prerequisite policy that the exam metadata defines."""
    problems = exam.get("problems")
    if not problems:
        return "Đề một câu."
    lines = [describe_problem(p) for p in problems]
    lines.append("Chính sách: " + exam.get("grading_policy", ""))
    return "\n".join(lines)


def describe_problem(problem: dict) -> str:
    """Describe one problem using only the fields it defines (function exams vs stdin/stdout exams)."""
    line = f"{problem['pid']}: tối đa {problem['max_score']} điểm"
    if problem.get("prototype"):
        line += f", hàm `{problem['prototype']}`"
    if problem.get("io_mode"):
        line += f", vào/ra {problem['io_mode']}"
    if problem.get("constraints"):
        line += f", ràng buộc: {problem['constraints']}"
    return line


def build_user_prompt(sample_input: dict, exam: dict, code: str, code_chars: int) -> str:
    """Compose the grading prompt from exam, build output and code only; no teacher feedback."""
    return "\n\n".join([
        INSTRUCTION,
        f"Loại đề: {sample_input['exam_type']}",
        "## Đề bài\n" + truncate_middle(exam["statement"], MAX_STATEMENT_CHARS),
        "## Cấu trúc điểm\n" + describe_grading_policy(exam),
        "## Kết quả biên dịch\n"
        + (truncate_middle(sample_input.get("compile_log") or "", MAX_COMPILE_LOG_CHARS) or "(không có lỗi)"),
        "## Kết quả test\n" + summarize_test_report(sample_input.get("test_report")),
        "## Mã nguồn C++\n```cpp\n" + truncate_middle(code, code_chars) + "\n```",
    ])


def build_target_text(output: dict) -> str:
    return json.dumps(output["rubric"], ensure_ascii=False)


def fit_prompt_ids(tokenizer, sample_input, exam, code, max_prompt_tokens):
    """Shrink the code window until the chat-templated prompt fits the token budget."""
    code_chars = INITIAL_CODE_CHARS
    while True:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(sample_input, exam, code, code_chars)},
        ]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        if len(ids) <= max_prompt_tokens or code_chars <= MIN_CODE_CHARS:
            return ids
        code_chars //= 2
