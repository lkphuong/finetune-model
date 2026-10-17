"""Task 1 metrics: QWK on total score, MAE, and per-dimension exact match."""
from sklearn.metrics import cohen_kappa_score

from config import RUBRIC_MAX

TOTAL_SCORE_LABELS = list(range(11))


def grading_metrics(gold: list[dict], pred: list[dict]) -> dict:
    g_total = [x["total_score"] for x in gold]
    p_total = [x["total_score"] for x in pred]
    metrics = {
        "qwk": cohen_kappa_score(g_total, p_total, weights="quadratic", labels=TOTAL_SCORE_LABELS),
        "mae": sum(abs(a - b) for a, b in zip(g_total, p_total)) / len(g_total),
    }
    for name in RUBRIC_MAX:
        metrics[f"exact_{name}"] = sum(g["rubric"][name] == p["rubric"][name] for g, p in zip(gold, pred)) / len(gold)
    return metrics
