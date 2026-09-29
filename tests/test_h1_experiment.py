"""Tests for the H1 experiment scaffold."""

from __future__ import annotations

from phailogeny.analysis.h1_experiment import h1_summary


def test_h1_summary_reports_delta_between_models() -> None:
    """The experiment helper should summarize performance differences between M0 and M1."""
    results = [{"m0": 0.80, "m1": 0.70}, {"m0": 0.75, "m1": 0.60}]
    summary = h1_summary(results)

    assert abs(summary["mean_m0"] - 0.775) < 1e-9
    assert abs(summary["mean_m1"] - 0.65) < 1e-9
    assert abs(summary["mean_delta"] + 0.125) < 1e-9
