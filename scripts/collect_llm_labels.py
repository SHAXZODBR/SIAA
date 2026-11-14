#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — STAGE 2b "second read": COLLECT THE LLM LABELS OF A REPORT BATCH
================================================================================
  WHAT
    The rules of scripts/label_reports.py label every brain conclusion in
    seconds. This script adds an independent SECOND READ by an LLM and keeps a
    label only where the readers agree:

      round 1  one LLM read of EVERY brain row (training/relabel_workflow.js,
               direct mode, votes: 1). A row is DECIDED when the LLM label
               equals the rule label. Everything else is DISPUTED: the export
               line is copied verbatim to --disputed.
      round 2  three LLM reads of the disputed rows (same workflow, votes: 3).
               The majority decides the abnormal bit, then the most common
               finding among the majority votes (tie → the rule label when it
               is one of the tied findings, else the most severe one). A row
               whose three reads do not agree on normal-vs-abnormal is a
               borderline report: it gets the majority label but
               needs_review=yes, so stage "build" keeps it out of the training
               set (--skip-needs-review).

    The output (--out) is a JSONL for

        scripts/label_reports.py --import-llm-labels <out>
        scripts/release_pipeline.py … --llm-labels <out>

  WHY IT IS SAFE
    * Agents write `<row>\\t<patient_id>\\t<label>` lines to
      <export>.labels/v<vote>-b<batch>.tsv. A vote is accepted only when the row
      is inside the export, the patient_id equals the export line's and the
      label is one of the nine. Everything else is REJECTED and counted — a
      rejected vote can only send a row to the next round, never label it.
    * With --result (the workflow's return value) every vote is cross-checked
      against the part files of its out_dir; a mismatch rejects that vote.
      When that run left no part file at all, the --result rows are used instead.
    * The four import keys (patient_id, study_date, modality, section_index)
      are copied from the EXPORT, never from an agent's answer.
    * More than --max-rejected (default 2 %) rejected votes → exit 3: an agent
      mis-aligned its rows, re-run the workflow — or, with row_out_of_range
      rejections, part files of an earlier run are left in the parts directory:
      delete it, then re-run the workflow. Exit 3 prints the rejected count per
      part file.

  USAGE (from sentinel/; PY=/Users/shakhzodbtr/venv/bin/python)
    # export every brain row of the batch once; the RULE labels go to their own file (the
    # import below must not overwrite it: --rules refuses a file that holds imported labels)
    $PY scripts/label_reports.py --reports data/<batch>_reports.csv --out data/<batch>_labels_rules.csv \\
        --export-for-llm data/<batch>_review_all.jsonl --all
    # round 1 — after Workflow({scriptPath:'training/relabel_workflow.js',
    #            args:{file:'<abs>/data/<batch>_review_all.jsonl', n_lines:<wc -l>, votes:1}})
    $PY scripts/collect_llm_labels.py --export data/<batch>_review_all.jsonl --rules data/<batch>_labels_rules.csv \\
        --out data/<batch>_llm_round1.jsonl --disputed data/<batch>_disputed.jsonl
    # round 2 — after the same Workflow on data/<batch>_disputed.jsonl with votes: 3
    $PY scripts/collect_llm_labels.py --export data/<batch>_disputed.jsonl --rules data/<batch>_labels_rules.csv \\
        --prior data/<batch>_llm_round1.jsonl --out data/<batch>_llm_labels.jsonl
    # import (release_pipeline.py does this itself with --llm-labels data/<batch>_llm_labels.jsonl)
    $PY scripts/label_reports.py --reports data/<batch>_reports.csv --out data/<batch>_labels.csv \\
        --import-llm-labels data/<batch>_llm_labels.jsonl

    Part files are read from <export>.labels/ (where the workflow writes them by
    default; --parts overrides). Start EACH round with an empty (or absent)
    parts directory: every v<vote>-b<batch>.tsv in it is read, and a workflow
    run overwrites only the batches it needs, so part files of an earlier run
    (e.g. of a longer disputed file) stay and are counted as rejected votes
    (exit 3 lists them). Without --disputed the run is FINAL: a row that
    still has no decision keeps its rule label and is written with
    needs_review=yes; a row with no rule label either is left out (counted).

  EXIT CODES
    0 ok · 2 usage / unreadable input / blank export line / duplicate export keys /
    a --rules file that already holds imported labels / a row labelled both in
    --prior and in this round / no vote lines at all (nothing is written) ·
    3 too many rejected votes, including "every vote rejected" (only --summary is
    written: --out and --disputed are left untouched, so nothing can be imported
    from a mis-aligned run)

  PHI: prints counts only. The disputed file holds the same boilerplate-stripped
  conclusions as the export and stays under data/ (git-ignored).
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_reports as L  # noqa: E402

G, Y, R, B, NC = L.G, L.Y, L.R, L.B, L.NC

PART_RX = re.compile(r'^v(\d+)-b(\d+)\.tsv$')
KEY_FIELDS = ('patient_id', 'study_date', 'modality', 'section_index')
MAJORITY_VOTES = 3            # fewer votes than this can only CONFIRM the rule label
REJECT_REASONS = ('malformed', 'row_out_of_range', 'patient_mismatch', 'bad_label',
                  'conflicting_duplicate', 'source_mismatch')


class Decision(NamedTuple):
    label: str
    how: str            # agree | unanimous | majority | tie_rules | tie_severity
    review: bool        # True → imported with needs_review=yes (kept out of the training set)


class InputError(Exception):
    """Unusable input (exit 2)."""


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def read_export(path: Path) -> tuple[list[dict], list[str]]:
    """The --export-for-llm JSONL: (parsed rows, raw lines), one per line, in file order.

    The 0-based position in this list is the `row` the workflow agents answer with
    (they print the file with `awk '{print NR-1 …}'`), so blank lines are an error
    rather than silently skipped.
    """
    rows, raw = [], []
    try:
        with open(path, encoding='utf-8') as fh:
            lines = list(fh)
    except UnicodeDecodeError as exc:
        raise InputError(f'{path}: not UTF-8 ({exc.reason})') from None
    for n, line in enumerate(lines, 1):
        text = line.rstrip('\n').rstrip('\r')
        if not text.strip():
            raise InputError(f'{path}: line {n} is blank — row numbers would shift')
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(f'{path}: line {n} is not JSON ({exc.msg})') from None
        if not isinstance(obj, dict) or not str(obj.get('patient_id', '')).strip():
            raise InputError(f'{path}: line {n} has no patient_id')
        rows.append(obj)
        raw.append(text)
    keys = Counter(L.row_key(r) for r in rows)
    dup = sum(1 for c in keys.values() if c > 1)
    if dup:
        raise InputError(f'{path}: {dup} key(s) (patient_id, study_date, modality, section_index) '
                         f'occur more than once — the import could not tell the rows apart')
    return rows, raw


class Votes:
    """Accepted votes per row, plus what was rejected and why."""

    def __init__(self, export: list[dict]):
        self.export = export
        self.by_vote: dict[tuple, dict[int, str]] = defaultdict(dict)   # (run, vote) → {row: label}
        self.rejected: Counter = Counter()
        self.seen = 0                                                   # vote lines read, accepted or not
        self.part_rejections: list[tuple[Path, str, Counter]] = []      # (dir, part file name, rejected)

    def add(self, run, vote: int, row, patient_id, label) -> None:
        self.seen += 1
        try:
            i = int(str(row).strip())
        except ValueError:
            self.rejected['malformed'] += 1
            return
        if not 0 <= i < len(self.export):
            self.rejected['row_out_of_range'] += 1
            return
        if str(patient_id).strip() != str(self.export[i].get('patient_id', '')).strip():
            self.rejected['patient_mismatch'] += 1
            return
        lab = str(label).strip().lower()
        if lab not in L.LABELS:
            self.rejected['bad_label'] += 1
            return
        slot = self.by_vote[(run, vote)]
        if i in slot:
            if slot[i] is not None and slot[i] != lab:
                self.rejected['conflicting_duplicate'] += 2     # neither answer can be trusted
                slot[i] = None
            return
        slot[i] = lab

    def cross_check(self, run, other: 'Votes', other_run) -> None:
        """Drop every vote of `run` that the second source (`other`) reports differently."""
        for (r, vote), slot in self.by_vote.items():
            if r != run:
                continue
            ref = other.by_vote.get((other_run, vote), {})
            for i, lab in list(slot.items()):
                if lab is None:
                    continue
                if i in ref and ref[i] is not None and ref[i] != lab:
                    self.rejected['source_mismatch'] += 1
                    slot[i] = None

    def for_row(self, i: int) -> list[str]:
        out = []
        for key in sorted(self.by_vote):
            lab = self.by_vote[key].get(i)
            if lab is not None:
                out.append(lab)
        return out

    @property
    def n_votes(self) -> int:
        return len(self.by_vote)

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected.values())


def read_parts(votes: Votes, run, directory: Path) -> int:
    """Add every v<vote>-b<batch>.tsv of one workflow run. Returns the number of part files."""
    files = sorted((p for p in directory.iterdir() if PART_RX.match(p.name)),
                   key=lambda p: tuple(int(x) for x in PART_RX.match(p.name).groups()))
    for p in files:
        vote = int(PART_RX.match(p.name).group(1))
        before = Counter(votes.rejected)
        for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                continue
            cells = line.split('\t')
            if len(cells) != 3:
                votes.seen += 1
                votes.rejected['malformed'] += 1
                continue
            votes.add(run, vote, *cells)
        delta = votes.rejected - before
        if delta:                                   # for the exit-3 diagnosis (file names carry no PHI)
            votes.part_rejections.append((directory, p.name, delta))
    return len(files)


def read_result(path: Path) -> dict:
    """The workflow's return value (or the task output file that wraps it in {"result": …})."""
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f'{path}: not a readable JSON file ({exc})') from None
    if isinstance(obj, dict) and isinstance(obj.get('result'), dict):
        obj = obj['result']
    if not isinstance(obj, dict) or not isinstance(obj.get('rows'), list):
        raise InputError(f'{path}: no "rows" list — expected the return value of '
                         f'training/relabel_workflow.js in direct mode')
    return obj


def add_result(votes: Votes, run, result: dict) -> None:
    for line in result['rows']:
        cells = str(line).split('\t')
        if len(cells) != 4 or not cells[0].strip().isdigit():
            votes.seen += 1
            votes.rejected['malformed'] += 1
            continue
        votes.add(run, int(cells[0]), *cells[1:])


def load_rules(path: Path) -> dict[tuple, dict]:
    """Brain rows of the RULES-ONLY stage-2 labels CSV by import key. A key that occurs twice is
    unusable; a file whose brain rows carry imported / LLM labels is refused (InputError)."""
    try:
        rows, _ = L.read_csv(path)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise InputError(f'{path}: not a readable CSV file ({exc})') from None
    out: dict[tuple, Optional[dict]] = {}
    not_rules = 0
    for r in rows:
        if (r.get('body_part') or '').strip().lower() != 'brain':
            continue
        if (r.get('label_source') or '').strip().lower() not in ('', 'rules'):
            not_rules += 1
        k = L.row_key(r)
        out[k] = None if k in out else r
    if not_rules:
        # the labels CSV was overwritten by an import (label_reports.py --import-llm-labels / --llm
        # with the same --out): a second read compared with it would "agree" with earlier LLM labels
        raise InputError(f'{path}: {not_rules} brain row(s) already carry imported / LLM labels '
                         f'(label_source is not "rules"). The second read must be compared with the RULE '
                         f'labels: write them to their own file — label_reports.py --reports <reports csv> '
                         f'--out <rules csv> — and pass that file as --rules')
    return {k: r for k, r in out.items() if r is not None}


# ----------------------------------------------------------------------------
# Deciding
# ----------------------------------------------------------------------------
def decide(votes: list[str], rule: str) -> Optional[Decision]:
    """One row: the accepted LLM votes + the rule label ('' when the rules have none).

    < 3 votes  a second read can only CONFIRM: decided when every vote equals the rule label.
    ≥ 3 votes  adjudication: majority on normal-vs-abnormal, then the most common finding among
               the majority votes. A split on normal-vs-abnormal still gets the majority label but
               review=True. None = undecided.
    """
    n = len(votes)
    if n == 0:
        return None
    if n < MAJORITY_VOTES:
        if rule in L.LABELS and all(v == rule for v in votes):
            return Decision(rule, 'agree', False)
        return None
    n_abn = sum(1 for v in votes if v != 'normal')
    if n_abn * 2 == n:                                   # even number of votes, split down the middle
        if rule not in L.LABELS:
            return None
        abnormal = rule != 'normal'
    else:
        abnormal = n_abn * 2 > n
    pool = Counter(v for v in votes if (v != 'normal') == abnormal)
    top = max(pool.values())
    tied = [lab for lab, c in pool.items() if c == top]
    split = n_abn not in (0, n)
    if len(tied) == 1:
        how = 'unanimous' if len(pool) == 1 and not split else 'majority'
        return Decision(tied[0], how, split)
    if rule in tied:
        return Decision(rule, 'tie_rules', split)
    return Decision(min(tied, key=L.SEVERITY_RANK.__getitem__), 'tie_severity', split)


def _out_line(exp: dict, label: str, how: str, review: bool, votes: list[str], rule: str) -> dict:
    line = {k: '' if exp.get(k) is None else str(exp.get(k)) for k in KEY_FIELDS}     # keeps a numeric 0
    line.update(label=label, abnormal=label != 'normal', needs_review='yes' if review else 'no',
                decided_by=how, votes=list(votes), rule=rule)
    return line


def collect(export: list[dict], raw: list[str], votes: Votes, rules: dict[tuple, dict],
            final: bool) -> dict:
    """→ {'out': [label lines], 'disputed': [raw export lines], 'stats': {...}}."""
    out, disputed = [], []
    how_counts: Counter = Counter()
    why_disputed: Counter = Counter()
    n_votes_hist: Counter = Counter()
    pairs: Counter = Counter()                      # (rule label, first LLM vote) where they differ
    cmp_n = cmp_label = cmp_abn = 0
    flips = Counter()                               # rule→llm on the abnormal bit
    n_review = n_unresolved = n_dropped = n_no_rule = 0
    for i, exp in enumerate(export):
        rule_row = rules.get(L.row_key(exp))
        rule = (rule_row or {}).get('label', '')
        rule = rule if rule in L.LABELS else ''
        if not rule:
            n_no_rule += 1
        v = votes.for_row(i)
        n_votes_hist[len(v)] += 1
        if v and rule:
            cmp_n += 1
            cmp_label += v[0] == rule
            same_bit = (v[0] != 'normal') == (rule != 'normal')
            cmp_abn += same_bit
            if v[0] != rule:
                pairs[(rule, v[0])] += 1
            if not same_bit:
                flips['normal→abnormal' if rule == 'normal' else 'abnormal→normal'] += 1
        d = decide(v, rule)
        if d is not None:
            out.append(_out_line(exp, d.label, d.how, d.review, v, rule))
            how_counts[d.how] += 1
            n_review += d.review
            continue
        if not v:
            why = 'no_vote'
        elif not rule:
            why = 'no_rule_label'
        elif len(v) < MAJORITY_VOTES and len(set(v)) > 1:
            why = 'votes_differ'
        elif len(v) < MAJORITY_VOTES:
            why = 'abnormal_bit_differs' if (v[0] != 'normal') != (rule != 'normal') else 'finding_differs'
        else:
            why = 'tied_votes'
        why_disputed[why] += 1
        if not final:
            disputed.append(raw[i])
        elif rule:
            out.append(_out_line(exp, rule, 'unresolved', True, v, rule))
            n_unresolved += 1
        else:
            n_dropped += 1
    stats = {
        'n_export': len(export), 'votes_per_row': dict(sorted(n_votes_hist.items())),
        'n_vote_sets': votes.n_votes, 'votes_seen': votes.seen,
        'rejected': {k: votes.rejected[k] for k in REJECT_REASONS if votes.rejected[k]},
        'compared': cmp_n,
        'label_agreement': round(cmp_label / cmp_n, 4) if cmp_n else None,
        'abnormal_agreement': round(cmp_abn / cmp_n, 4) if cmp_n else None,
        'abnormal_flips': dict(flips),
        'top_disagreements': [[a, b, c] for (a, b), c in pairs.most_common(12)],
        'decided': sum(how_counts.values()), 'decided_by': dict(how_counts), 'review': n_review,
        'undecided': sum(why_disputed.values()), 'undecided_why': dict(why_disputed),
        'disputed_written': len(disputed), 'unresolved_kept_rule_label': n_unresolved,
        'dropped_no_label': n_dropped, 'no_rule_label': n_no_rule, 'final': final,
        'labels': dict(Counter(o['label'] for o in out)),
    }
    return {'out': out, 'disputed': disputed, 'stats': stats}


def read_prior(path: Path) -> list[dict]:
    try:
        lines = L.read_jsonl(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f'{path}: not a readable JSONL file ({exc})') from None
    for n, lab in enumerate(lines, 1):
        if not isinstance(lab, dict) or str(lab.get('label', '')).strip().lower() not in L.LABELS:
            raise InputError(f'{path}: entry {n} has no valid label')
    return lines


# ----------------------------------------------------------------------------
def print_stats(s: dict) -> None:
    print('\n' + '=' * 72)
    print(f'  {B}LLM SECOND READ — {s["n_export"]} exported rows, {s["n_vote_sets"]} vote set(s){NC}')
    print('=' * 72)
    print(f'  votes per row      : {s["votes_per_row"]}')
    rej = sum(s['rejected'].values())
    print(f'  votes read         : {s["votes_seen"]}   rejected: {Y if rej else G}{rej}{NC} {s["rejected"] or ""}')
    if s['compared']:
        print(f'  LLM vs rules ({s["compared"]} rows, first vote): abnormal agreement '
              f'{s["abnormal_agreement"]:.3f}, label agreement {s["label_agreement"]:.3f}')
        if s['abnormal_flips']:
            print(f'    normal-vs-abnormal disagreements (rules→LLM): {s["abnormal_flips"]}')
        for a, b, c in s['top_disagreements']:
            print(f'    {c:>5}  rules {a:<14} LLM {b}')
    print(f'  decided            : {G}{s["decided"]}{NC} {s["decided_by"]}')
    if s['review']:
        print(f'    of which needs_review=yes (readers split on normal-vs-abnormal): {Y}{s["review"]}{NC}')
    col = Y if s['undecided'] else G
    print(f'  undecided          : {col}{s["undecided"]}{NC} {s["undecided_why"] or ""}')
    if s['final']:
        if s['unresolved_kept_rule_label']:
            print(f'    kept the rule label with needs_review=yes: {s["unresolved_kept_rule_label"]}')
        if s['dropped_no_label']:
            print(f'    {Y}no vote and no rule label — left out: {s["dropped_no_label"]}{NC}')
    else:
        print(f'    → written to the disputed file for a {MAJORITY_VOTES}-vote round: {s["disputed_written"]}')


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Collect + verify the LLM labels written by training/relabel_workflow.js (direct mode).')
    ap.add_argument('--export', required=True, help='the JSONL the workflow read (label_reports.py --export-for-llm)')
    ap.add_argument('--parts', action='append', default=None,
                    help='directory with v<vote>-b<batch>.tsv part files (default <export>.labels; repeatable — '
                         'each directory is a separate set of votes)')
    ap.add_argument('--result', action='append', default=[],
                    help='the workflow return value as JSON (or the task output file). Cross-checks the part '
                         'files of its out_dir; without part files it is used instead (repeatable)')
    ap.add_argument('--rules', required=True, help='stage-2 labels CSV of the same batch (the rule labels)')
    ap.add_argument('--prior', action='append', default=[],
                    help='labels decided in an earlier round (copied into --out first; repeatable)')
    ap.add_argument('--out', required=True, help='JSONL for label_reports.py --import-llm-labels')
    ap.add_argument('--disputed', help='write undecided export lines here for the next round; '
                                       'omit on the FINAL round')
    ap.add_argument('--summary', help='optional JSON with the counts printed on screen')
    ap.add_argument('--max-rejected', type=float, default=0.02,
                    help='largest tolerated share of rejected votes (default 0.02)')
    a = ap.parse_args(argv)

    try:
        export_path = Path(a.export)
        for p in [export_path, Path(a.rules)] + [Path(x) for x in a.prior + a.result]:
            if not p.is_file():
                raise InputError(f'not found: {p}')
        export, raw = read_export(export_path)
        if not export:
            raise InputError(f'{export_path}: empty')
        rules = load_rules(Path(a.rules))

        explicit = a.parts is not None
        part_dirs = [Path(x) for x in (a.parts if explicit else [f'{a.export}.labels'])]
        votes = Votes(export)
        runs: dict[str, tuple] = {}                 # resolved dir → run id
        n_files = 0
        for n, d in enumerate(part_dirs):
            if not d.is_dir():
                if explicit or not a.result:
                    raise InputError(f'no part-file directory: {d}')
                continue
            k = read_parts(votes, ('parts', n), d)
            n_files += k
            if k:                                   # an empty directory leaves a --result usable directly
                runs[str(d.resolve())] = ('parts', n)
        for n, rp in enumerate(a.result):
            result = read_result(Path(rp))
            out_dir = str(result.get('out_dir') or '')
            target = runs.get(str(Path(out_dir).resolve())) if out_dir else None
            if target is None:
                add_result(votes, ('result', n), result)       # no part files for this run: use it directly
            else:
                check = Votes(export)
                add_result(check, 'check', result)
                votes.cross_check(target, check, 'check')
        if not votes.by_vote and not votes.seen:
            raise InputError('no votes found (no v<vote>-b<batch>.tsv part file and no --result rows)')

        prior: list[dict] = []
        for pp in a.prior:
            prior.extend(read_prior(Path(pp)))
        res = collect(export, raw, votes, rules, final=not a.disputed)
        seen = Counter(L.row_key(o) for o in prior + res['out'])
        overlap = sum(1 for c in seen.values() if c > 1)
        if overlap:
            raise InputError(f'{overlap} row(s) are labelled more than once across --prior and this round')
    except InputError as exc:
        print(f'{R}✗ {exc}{NC}')
        return 2

    s = res['stats']
    s['part_files'] = n_files
    s['prior'] = len(prior)
    s['total_out'] = len(prior) + len(res['out'])
    print_stats(s)

    share = votes.n_rejected / max(1, votes.seen)
    too_many = share > a.max_rejected
    s['rejected_share'] = round(share, 4)
    s['written'] = not too_many
    if a.summary:                                   # always: it is the diagnosis of a refused run too
        sp = Path(a.summary)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(s, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'  ✓ {sp}')
    if too_many:
        # an agent that shifted its rows by one can pass the patient check on a same-patient CT/MR
        # pair, so the ACCEPTED votes of such a run are not trustworthy either: write no labels
        print(f'\n  {R}✗ {votes.n_rejected} of {votes.seen} votes were rejected ({share:.1%} > '
              f'{a.max_rejected:.1%}) — an agent mis-aligned its rows. NOTHING was written to --out'
              f'{" / --disputed" if a.disputed else ""}; re-run the workflow, then this script{NC}')
        for d, name, rej in votes.part_rejections:
            print(f'    {d / name}: {sum(rej.values())} rejected {dict(rej)}')
        stale = sorted({str(d) for d, _, rej in votes.part_rejections if rej['row_out_of_range']})
        if stale:
            print(f'  {Y}⚠ row_out_of_range votes: part files of an earlier run (a longer export) may be '
                  f'left in the directory — a new run overwrites only the batches it needs, so re-running '
                  f'the workflow alone cannot fix this. Delete the directory, then re-run the workflow and '
                  f'this script:{NC}')
            for d in stale:
                print(f'      {d}')
        return 3

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        for line in prior + res['out']:
            fh.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'  ✓ {out}  ({len(prior)} from earlier rounds + {len(res["out"])} from this one)')
    if a.disputed:
        dp = Path(a.disputed)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(''.join(line + '\n' for line in res['disputed']), encoding='utf-8')
        print(f'  ✓ {dp}  ({len(res["disputed"])} rows for the next round)')
        if res['disputed']:
            print(f'    next: Workflow training/relabel_workflow.js with args.file = that file, '
                  f'n_lines = {len(res["disputed"])}, votes = {MAJORITY_VOTES}; then this script again with '
                  f'--export {dp} --prior {out} and no --disputed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
