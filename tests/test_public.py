import numpy as np
import pandas as pd
import pytest

from rawaj.audit import audit
from rawaj.metrics import (best_threshold, confusion_at, precision_at_k,
                           precision_recall_at, threshold_for_precision)


def small_extract():
    return pd.DataFrame({
        "txn_id": ["t1", "t2", "t3", "t3", "t5", "t6"],
        "sent_at": ["2026-09-01 10:00", "2026-09-01 10:05", "2026-09-02 09:00",
                    "2026-09-02 09:00", "not a date", "2026-09-03 23:59"],
        "amount_egp": [500.0, 12000.0, 800.0, 800.0, None, 50000.0],
        "channel": ["app", "app", " ", " ", "app", "ussd"],
        "currency": ["EGP"] * 6,
        "is_mule": [0, 1, 0, 0, None, 1],
    })


def test_audit_reports_basic_problems():
    r = audit(small_extract(), target="is_mule", id_col="txn_id", time_col="sent_at")
    assert r["n_rows"] == 6
    assert r["duplicate_ids"] == ["t3"]
    assert r["duplicate_rows"] == 1
    assert r["missing"]["channel"] == pytest.approx(2 / 6)   # blank strings count as missing
    assert r["unlabelled_rows"] == 1
    assert r["positive_rate"] == pytest.approx(2 / 5)
    assert r["bad_timestamps"] == 1
    assert "currency" in r["constant_columns"]


def test_confusion_counts_score_equal_to_threshold_as_flagged():
    y = [1, 0, 1, 0]
    s = [0.9, 0.5, 0.5, 0.1]
    assert confusion_at(y, s, 0.5) == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}


def test_precision_is_zero_when_nothing_is_flagged():
    p, r = precision_recall_at([1, 0, 0], [0.2, 0.1, 0.3], threshold=0.99)
    assert (p, r) == (0.0, 0.0)


def test_precision_at_k_uses_the_top_scores():
    y = [0, 1, 1, 0, 1]
    s = [0.1, 0.9, 0.8, 0.7, 0.2]
    assert precision_at_k(y, s, 2) == 1.0
    assert precision_at_k(y, s, 10) == pytest.approx(3 / 5)   # k larger than n uses n


def test_best_threshold_prefers_catching_mules_when_misses_are_expensive():
    y = [1, 0, 0, 0, 1, 0]
    s = [0.6, 0.55, 0.4, 0.3, 0.35, 0.1]
    t = best_threshold(y, s, cost_fp=300, cost_fn=20_000)
    assert t == pytest.approx(0.35)
    assert threshold_for_precision(y, s, 0.99) == pytest.approx(0.6)
