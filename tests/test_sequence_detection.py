"""Brain MR series-description → canonical sequence (src/pipeline/brain_mri_preprocessor.detect_sequence).

Regression for the 2026-11-09 clinic dress rehearsal: the Siemens diffusion series
'ep2d_diff_3scan_trace_p2_TRACEW' was classified 'unknown', so the report technique line omitted
DWI and the (experimental) stroke detector never found its preferred input. The anatomical
mapping the validated triage protocol depends on must not change.
"""
from __future__ import annotations

import pytest

from src.pipeline.brain_mri_preprocessor import detect_sequence


@pytest.mark.parametrize("descr,expected", [
    # diffusion: Siemens / GE / Philips naming
    ("ep2d_diff_3scan_trace_p2_TRACEW", "DWI"),
    ("ep2d_diff_3scan_trace_p2", "DWI"),
    ("AX DIFF", "DWI"),
    ("DWI_b1000", "DWI"),
    ("Diffusion 3 directions", "DWI"),
    # the derived ADC map is ADC, never DWI
    ("ep2d_diff_3scan_trace_p2_ADC", "ADC"),
    ("dDWI_ADC", "ADC"),
    ("ADC", "ADC"),
])
def test_diffusion_series(descr, expected):
    assert detect_sequence(descr) == expected


@pytest.mark.parametrize("descr,expected", [
    # anatomical sequences — unchanged (the triage input rule depends on them)
    ("t1_tse_sag", "T1"),
    ("t1_tse_sag_2.2 mm", "T1"),
    ("t2_tse_tra_512", "T2"),
    ("t2_tse_COR", "T2"),
    ("FLAIR_tra", "FLAIR"),
    ("T2_FLAIR", "FLAIR"),
    ("AX_T1_POST", "T1ce"),
    ("T1_axial", "T1"),
    # not a diagnostic sequence
    ("AAhead_scout (11 sec)", None),
    ("PhoenixZIPReport", None),
    ("", None),
    # 'diff'/'trace' must be whole tokens
    ("difference_map", None),
    ("tracer_uptake", None),
])
def test_other_series_unchanged(descr, expected):
    assert detect_sequence(descr) == expected
