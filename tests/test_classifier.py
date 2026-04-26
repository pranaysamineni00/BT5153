import sys
import types

import numpy as np
import pytest

import classifier


def _blank_classifier():
    clf = classifier.LegalClauseClassifier.__new__(classifier.LegalClauseClassifier)
    clf.mode = "heuristic"
    clf.model = None
    clf.tokenizer = None
    clf.id_to_clause = {}
    clf.thresholds = {}
    clf.calibration_temperature = 1.0
    clf.threshold_source = "heuristic_default"
    clf.reliability_status = "degraded"
    clf.reliability_warning = "heuristic fallback"
    clf._device = None
    clf._mode_preference = "auto"
    return clf


def test_load_tfidf_baseline_prefers_saved_per_clause_thresholds(monkeypatch):
    payload = {
        "model": object(),
        "id_to_clause": {0: "A", 1: "B"},
        "best_threshold": 0.4,
        "per_clause_thresholds": {0: 0.8, "1": 0.3},
        "calibration_temperature": 1.7,
    }
    monkeypatch.setitem(sys.modules, "joblib", types.SimpleNamespace(load=lambda path: payload))

    clf = _blank_classifier()
    clf._load_tfidf_baseline("dummy.joblib")

    assert clf.mode == "baseline"
    assert clf.original_thresholds == {"A": pytest.approx(0.8), "B": pytest.approx(0.3)}
    assert clf.thresholds == {"A": pytest.approx(0.55), "B": pytest.approx(0.35)}
    assert clf.threshold_source == "artifact_per_clause"
    assert clf.calibration_temperature == pytest.approx(1.7)
    assert clf.reliability_status == "ready"


def test_calibrate_probabilities_softens_extreme_scores():
    clf = _blank_classifier()
    clf.calibration_temperature = 2.0

    calibrated = clf._calibrate_probabilities(np.array([0.95, 0.05]))

    assert calibrated[0] < 0.95
    assert calibrated[0] > 0.5
    assert calibrated[1] > 0.05
    assert calibrated[1] < 0.5


def test_classify_reports_degraded_mode():
    clf = _blank_classifier()
    clf._classify = lambda text: [
        {"clause": "Non-Compete", "risk": "HIGH", "confidence": 0.61}
    ]

    result = clf.classify("sample text")

    assert result["degraded_mode"] is True
    assert result["reliability"]["status"] == "degraded"
    assert result["risk_summary"] == {"HIGH": 1, "MEDIUM": 0, "LOW": 0}


def test_classifier_mode_preference_defaults_to_baseline(monkeypatch):
    monkeypatch.delenv("LEXSCAN_CLASSIFIER_MODE", raising=False)

    assert classifier._classifier_mode_preference() == "baseline"
