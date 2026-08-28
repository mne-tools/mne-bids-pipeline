"""Test report helpers."""

import mne

from mne_bids_pipeline._report import _sort_run_sections


def _titles(report: mne.Report) -> list[str]:
    return report.get_contents()[0]


def test_sort_run_sections() -> None:
    """Test that run sections sort by run within their group, others stay put."""
    report = mne.Report(title="sub-01")
    # Parallel completion order: runs finished 03, 01, 02; unrelated sections mixed
    # in, plus a second group and a non-numeric run
    report.add_html("<p/>", title="Data quality", tags=("data-quality",))
    report.add_html("<p/>", title="Raw: run-03", tags=("raw", "run-03"))
    report.add_html("<p/>", title="Raw: run-01", tags=("raw", "run-01"))
    report.add_html("<p/>", title="Events", tags=("events",))
    report.add_html("<p/>", title="SSP: run-02", tags=("ssp", "run-02"))
    report.add_html("<p/>", title="Raw: run-02", tags=("raw", "run-02"))
    report.add_html("<p/>", title="SSP: run-01", tags=("ssp", "run-01"))
    report.add_html("<p/>", title="Raw: run-noise", tags=("raw", "run-noise"))

    _sort_run_sections(report)
    assert _titles(report) == [
        "Data quality",
        # The raw group gathers where its first member sat, numeric runs first
        "Raw: run-01",
        "Raw: run-02",
        "Raw: run-03",
        "Raw: run-noise",
        "Events",
        "SSP: run-01",
        "SSP: run-02",
    ]

    # Already-sorted content is left alone (idempotent)
    before = _titles(report)
    _sort_run_sections(report)
    assert _titles(report) == before
