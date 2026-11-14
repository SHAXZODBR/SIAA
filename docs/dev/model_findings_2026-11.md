# Brain triage model — what batch 2 (new1) taught us (November 2026)

Developer note. Everything here is measured on this Mac with the scripts named; no patient
identifiers. Numbers are study-level, shipped protocol (mean abnormal probability of the central
axial slices ≥ 0.50), unless a line says otherwise. Raw outputs: `data/release/brain-triage-v2/`
(git-ignored).

## 1. The shipped model on patients it never saw (`prospective` stage)

Model `0d559766ce58`, batch new1 (reports of 2026-06-27 … 2026-09-20), MR brain only, 518 studies of
512 patients that are in no earlier set:

| | value | 95 % CI |
|---|---|---|
| sensitivity | 0.913 | 0.88 – 0.94 |
| specificity | 0.336 | 0.26 – 0.42 |
| AUC | 0.776 | |
| studies flagged | 84.6 % | |
| TP / FN / TN / FP | 345 / 33 / 47 / 93 | |

Sensitivity holds (validated 0.90 on the 178 locked patients). Specificity is lower than the 0.47 of
the locked set: two of three normal studies are flagged, and 41 % of the studies that are *not*
flagged are abnormal. The product must keep saying what it says today: a flag is a prompt to look
first, and no flag is never "normal".

## 2. Release `brain-triage-v2`: retrained, gated, **not promoted**

Same recipe as release 1, now with batch 1 + batch 2 (1,083 train / 41 tune / 281 gate patients; the
gate set is the 178 locked patients + 103 new ones; 206 abnormal and 76 normal studies):

| model | sensitivity | specificity | balanced | TP / FN / TN / FP |
|---|---|---|---|---|
| shipped (`0d559766ce58`) | 0.888 | 0.447 | 0.668 | 183 / 23 / 34 / 42 |
| candidate warm | 0.850 | 0.474 | 0.662 | 175 / 31 / 36 / 40 |
| candidate fresh | 0.825 | 0.461 | 0.643 | 170 / 36 / 35 / 41 |

Both candidates fail the gate (sensitivity floor 0.85 and "not below the shipped model − 0.03"). The
pipeline stopped before `promote`; the shipped model is unchanged. Doubling the data with the same
recipe moved the operating point along the same curve and did not improve discrimination.

## 3. Why more of the same did not help

**The model mostly measures atrophy.** AUC of the shipped model, normal studies (140) against each
finding, new patients (`data/release/brain-triage-v2/notes/finding_auc_shipped_model.json`):

| finding | studies | AUC vs normal | flagged at 0.50 |
|---|---|---|---|
| stroke | 34 | 0.90 | 97 % |
| atrophy | 245 | 0.81 | 94 % |
| tumor | 16 | 0.79 | 100 % |
| other | 36 | 0.69 | 81 % |
| cyst | 13 | 0.63 | 85 % |
| MS / white-matter lesions | 31 | 0.55 | 71 % |
| *normal* | 140 | | 66 % |

Normal studies score 0.61 on average. Focal findings (white-matter lesions, cysts) are flagged about
as often as normal studies; they are not really detected. Splitting the same scores into "any
non-atrophy finding" against "normal or atrophy" gives AUC 0.53. Report labels are study-level, 58 %
of the abnormal studies are atrophy, and the input is five central slices at 224 px: the dominant,
global signal is what gets learned.

**The trainer memorises.** Without augmentation the unfrozen blocks reach training accuracy 1.00 by
epoch 6 while the selection loss doubles; what generalises is learned in the first epochs. On a
122-patient development split of new *training* patients (never the gate set): no augmentation, last
epoch AUC 0.790 (sens 0.91 / spec 0.49 at 0.50); with `--augment`, last epoch AUC 0.815, but the
operating point moves to sens 0.67 / spec 0.79. Augmentation is better and more stable, and it needs a
calibrated operating point before it can pass the gate. The trainer now has `--augment`,
`--select-metric study` and `--seed`; the pipeline still runs the release-1 recipe.

**The input sequence is not the same from study to study.** `triage_sequence` takes the first
T1ce / T1 / T2 / FLAIR series *by name* and uses it only if it is axial; when the first T2 series is
sagittal it falls through to FLAIR. On 40 batch-2 studies the triage slices came from FLAIR in 20,
T2 in 16, T1 in 2. Every study offers axial T2 and FLAIR, and DWI + ADC.

**But the input sequence is not the limit either.** Three-fold patient-level cross-validation on
batch 2 alone (553 studies of 546 patients, 403 abnormal / 150 normal; new model per fold, with
augmentation, last epoch, no checkpoint selection on the test fold; every arm uses the same patients in
the same folds):

| input | AUC, mean of folds | per fold |
|---|---|---|
| today's rule (FLAIR or T2, whichever the rule picks) | 0.793 | 0.789 / 0.824 / 0.766 |
| always axial FLAIR | 0.797 | 0.786 / 0.805 / 0.801 |
| FLAIR + T2 as colour channels | 0.790 | 0.775 / 0.790 / 0.804 |

No difference. And a model trained on about 365 batch-2 patients reaches the same AUC (0.79) as the
shipped model trained on 715 batch-1 patients (0.78 on the same kind of patients) and as the
candidates trained on 1,083. With this design — five central slices, 224 px, one study-level label
dominated by atrophy — discrimination sits at about 0.8 whatever the amount of data, the recipe or the
sequence. More of the same data will not lift it; a different design can (section 5).
(`data/release/brain-triage-v2/notes/sequence_experiment_cv.json`. DWI was left out: the DWI series
holds several b-values and another field of view and needs its own handling.)

## 4. Head CT: a first local model

The public hemorrhage model that sits behind the *pending* head-CT route found 1 of 9 reported
hemorrhages in the batch-2 pre-read, and 12 of 234 head CT studies were routed as chest CT (the
series routing has been fixed since). A local model is the only way forward for CT.

First attempt, exploratory, not shipped: 197 labelled head CT studies of batch 2 (92 abnormal /
105 normal by the second-read report labels), nine central brain-window slices per study from the
archived sample (`src/pipeline/ct_head_selection.py`), new ViT with augmentation, five-fold
patient-level cross-validation, last epoch:

| | value |
|---|---|
| AUC, mean of folds | 0.746 (0.861 / 0.705 / 0.709 / 0.636 / 0.820) |
| sensitivity / specificity at 0.50 | 0.61 / 0.77 |

There is signal, from about 157 training studies per fold, and it is far from usable. What it needs:
several hundred more labelled studies (the second report delivery adds about 100), slices chosen by
anatomy rather than by position (in a head-and-neck acquisition the "central" slices are in the
neck), more than one window, and its own locked holdout.
(`data/release/brain-triage-v2/notes/head_ct_experiment_cv.json`)

## 5. What follows from this

1. **Keep the shipped model** for the pilot; quote section 1 next to the validated numbers.
2. **Never return a clinic drive before archiving it** (`scripts/intake_batch.py`, or
   `scripts/archive_studies.py` when the archive disk is smaller than the delivery). The raw images of
   batch 1 were not kept: batch 1 cannot be re-processed with another slice rule or sequence, and the
   178 locked patients cannot follow a protocol change. Batch 2 is archived under
   `~/SIAA_DATA/raw/shox/new1` (all MR complete, CT as a spaced sample per series).
3. **Next model generation** (new protocol, new locked holdout, both from archived raw data): the
   gain has to come from the design, not from more data of the same kind — attention over all slices
   of a series instead of five central ones, higher input resolution, targets that are not dominated
   by atrophy (at least a separate "focal finding" output, trained on the studies whose report names
   one), augmentation and a calibrated operating point. The images of 2026-09-16 … 2026-11-08 (reports already here: 528 MR brain
   studies, about 152 normal) are the next training and test material.
4. **Other body parts.** Normal-vs-abnormal only works where normal exists: about half of the head CT
   reports are normal, while about 99 % of spine, knee and joint MR reports describe findings. Those
   modules need finding-specific targets agreed with a radiologist before any training.
