"""A real GE/Siemens MRI study can contain 1,500-5,000 single-frame DICOM files.
Starlette's default multipart limit (1000 parts) must not reject it."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMALL_DCM = ROOT / 'data' / 'test_dicoms' / 'brain' / 'brain_normal_000.dcm'


def test_study_with_more_than_1000_files_is_not_rejected_by_part_limit(client):
    if not SMALL_DCM.exists():
        import pytest; pytest.skip('fixture DICOM missing')
    payload = SMALL_DCM.read_bytes()
    n = 1200
    files = [('files', (f'slice_{i:04d}.dcm', payload, 'application/dicom')) for i in range(n)]
    res = client.post('/analyze/study?language=ru', files=files)
    assert not (res.status_code == 400 and 'Too many files' in (res.text or '')), res.text[:200]
    assert res.status_code in (200, 422), res.text[:200]
    if res.status_code == 200:
        assert res.json().get('num_files') == n
