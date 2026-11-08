#!/usr/bin/env python3
"""Check that the restored validated-model bundle IS the release the code quotes.

WHAT: compares the release record that ships inside the installer
(models/brain_triage_finetuned/study_level_eval.json + MANIFEST.json, read with
src/inference/validated_metrics.load exactly as the installed app reads it) with the
release the code pins (validated_metrics.FALLBACK_*, rewritten by
scripts/bump_model_release.py and quoted by the intended-use / user-manual docs):

    identity  evaluated_on  tp  fn  tn  fp  n_studies  n_slices

WHY: the installer's model comes from the hand-set MODEL_BUNDLE_URL secret, the
numbers and dates the documents quote come from the committed bump. A re-evaluation
(new evaluated_at / protocol block) keeps the weights' identity, so an identity-only
check passes while the installer would show one evaluation date in the app (from the
bundled eval) and another in the shipped documents. Re-publish the bundle
(packaging/publish_model_bundle.py) after every bump.

    python packaging/check_bundle_release.py --model-dir models/brain_triage_finetuned

Exit codes: 0 = the bundle matches the code pin; 1 = mismatch (each differing field is
printed, with a GitHub Actions ::error annotation); 2 = no readable study_level_eval.json
in --model-dir (nothing to compare — the installer would fall back to the pinned figures
without a release record).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_UNREADABLE = 2

FIELDS = ('identity', 'evaluated_on', 'tp', 'fn', 'tn', 'fp', 'n_studies', 'n_slices')


def code_pin() -> dict:
    """The release the code quotes (validated_metrics.FALLBACK_*)."""
    from src.inference import validated_metrics as vm
    return {
        'identity': vm.FALLBACK_IDENTITY, 'evaluated_on': vm.FALLBACK_EVALUATED_ON,
        'tp': vm.FALLBACK_TP, 'fn': vm.FALLBACK_FN, 'tn': vm.FALLBACK_TN, 'fp': vm.FALLBACK_FP,
        'n_studies': vm.FALLBACK_N_STUDIES, 'n_slices': vm.FALLBACK_N_SLICES,
    }


def bundle_record(model_dir: Path) -> dict | None:
    """The release record in model_dir as the installed app reads it, or None when
    study_level_eval.json is missing/unreadable (validated_metrics fell back)."""
    from src.inference import validated_metrics as vm
    m = vm.load(Path(model_dir))
    if m.source != 'json':
        return None
    return {f: getattr(m, f) for f in FIELDS}


def compare(model_dir: Path) -> tuple[int, list[str]]:
    """(exit code, human lines) — pure, for tests."""
    pin = code_pin()
    rec = bundle_record(model_dir)
    if rec is None:
        return EXIT_UNREADABLE, [f'no readable study_level_eval.json in {model_dir} — cannot verify the release']
    lines = [f'{f:<13} bundle {rec[f]!s:<14} code {pin[f]!s}' for f in FIELDS]
    bad = [f for f in FIELDS if str(rec[f]) != str(pin[f])]
    if bad:
        lines.append('MISMATCH: ' + ', '.join(f'{f} (bundle {rec[f]} != code {pin[f]})' for f in bad))
        return EXIT_MISMATCH, lines
    return EXIT_OK, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Verify the restored model bundle is the release the code pins')
    ap.add_argument('--model-dir', default=str(REPO / 'models' / 'brain_triage_finetuned'),
                    help='restored validated model dir (default: models/brain_triage_finetuned)')
    args = ap.parse_args(argv)
    code, lines = compare(Path(args.model_dir))
    for line in lines:
        print(line)
    if code == EXIT_MISMATCH:
        print('::error title=Bundle release mismatch::The restored bundle is not the release the code '
              'and documents quote (see MISMATCH above). Re-publish the bundle after '
              'scripts/bump_model_release.py (packaging/publish_model_bundle.py) and point '
              'MODEL_BUNDLE_URL at it, or commit the bump for this bundle.')
    elif code == EXIT_UNREADABLE:
        print('::error title=Bundle release record missing::' + lines[0])
    else:
        print('bundle release record matches the code pin')
    return code


if __name__ == '__main__':
    sys.exit(main())
