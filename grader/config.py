"""Central constants for the C++ rubric grading pipeline (Task 1)."""
from dataclasses import dataclass

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MANIFEST_NAME = "run_manifest.json"
SUBMISSION_NAME = "submission_task1.json"
GRADING_FILE = "task1_grading.json"

RUBRIC_MAX = {
    "compilable": 1,
    "io_format": 1,
    "logic": 4,
    "edge_case": 2,
    "complexity": 1,
    "code_quality": 1,
}
# Dimensions that measure correctness of the program body; zeroed when the program cannot earn them
SOLUTION_DIMENSIONS = ("logic", "edge_case", "complexity", "code_quality")

MAX_STATEMENT_CHARS = 3000
MAX_COMPILE_LOG_CHARS = 1000
MAX_FAILED_TESTS_SHOWN = 5
INITIAL_CODE_CHARS = 8000
MIN_CODE_CHARS = 1000


@dataclass(frozen=True)
class TrainConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 5
    grad_accum_steps: int = 8
    max_seq_len: int = 4096
    max_new_tokens: int = 64
