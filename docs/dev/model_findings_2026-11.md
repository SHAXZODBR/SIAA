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
sequence. More of the same data will not lift it. Section 6 shows the reason, found a few hours later: patient age alone reaches the same 0.78, so the limit is the target, not the number of slices.
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
3. **Next model generation: another target, not only another design** (sections 6 and 7). Report
   abnormal-vs-normal mostly encodes age-typical change, and reading all slices does not improve it. The
   target that matters for triage is a *priority finding*; the first all-slice experiment ranks it clearly
   better than the shipped model, from archived raw data only. Before any release: confirm on the
   images of 2026-09-16 … 2026-11-08 (reports already here: 528 MR brain studies, about 152 normal),
   get radiologists' one-click priority labels, and judge every model by age-matched metrics and by
   age band.
4. **Other body parts.** Normal-vs-abnormal only works where normal exists: about half of the head CT
   reports are normal, while about 99 % of spine, knee and joint MR reports describe findings. Those
   modules need finding-specific targets agreed with a radiologist before any training.

## 6. The score follows patient age (measured 2026-11-14)

Same 518 never-seen studies, shipped model, shipped protocol. Age from the DICOM header
(`data/new1_index.csv`; the age tag agrees with birth-date arithmetic).

| abnormal vs normal | AUC |
|---|---|
| shipped model | 0.776 |
| **patient age alone** | **0.779** |
| difference, model − age (patient bootstrap) | −0.003, 95 % CI −0.041 … +0.036 |
| model, inside pairs of similar age (≤ 5 years apart; 7,175 pairs) | 0.619 (95 % CI 0.56 – 0.68) |
| logistic model on age / on age + model score (5-fold by patient) | 0.776 / 0.794 |

| age | abnormal studies flagged | normal studies flagged | sensitivity | specificity |
|---|---|---|---|---|
| under 40 | 103 / 132 | 64 / 110 | 0.78 | 0.42 |
| 40 – 59 | 139 / 142 | 29 / 30 | 0.98 | 0.03 |
| 60 + | 103 / 104 | no normal report exists | 0.99 | — |

The rule "flag everyone aged 20 or more" gives sensitivity 0.913 and specificity 0.271; the model at
0.50 gives 0.913 and 0.336. The model's score rises with age inside the normal class too (Spearman
0.60). It adds a small, real signal to age (the last table row; likelihood-ratio p < 0.001), not more.
The first head-CT model of section 4 is in the same position: on the 207 labelled head CT studies
(95 abnormal, 112 normal) age alone gives AUC 0.77, the model 0.75.

**Why: the reference.** 247 of the 258 "atrophy" conclusions — the largest abnormal class — contain
only age-typical wording: cerebral microangiopathy with a Fazekas grade (grade I in 127 of the 148
graded), dyscirculatory encephalopathy, moderate atrophy. Normal reports come from young patients
(median age 29) and none exists at 60 or more; in head CT of the same batch 14 do, so this is a
reporting habit for MR, not a law. At the same age the report template (a proxy for the reporting
radiologist; there is no author field) changes the odds of "normal" about two-fold. Inside pairs of
similar age the model separates tumour from normal at 0.53 (16 studies) and stroke at 0.52 (34): the
raw 0.79 and 0.90 of section 3 are mostly age. A re-read of 90 conclusions found the labels follow the
labelling rubric (88 of 90) and that none of 40 "atrophy" and none of 20 "normal" studies held a
finding that should move the read up the list, against 5 to 9 of 30 in the focal classes.

**What it changes.** The explanation at the end of section 3 was incomplete: five central slices are
not the limit, the *target* is. A model trained to tell report-abnormal from report-normal is paid for
recognising an older brain. Sensitivity 0.90 / specificity 0.47 remain true as measured and say little
about detection. The intended-use statements, the one-pagers, the pilot proposal and the pilot
protocol (age band and priority finding in the adjudication log, results by age band, comparison with
the rule "aged 40 or more") carry this since 2026-11-14; `docs/test_docs_consistency.py` pins it.

**Checked independently.** Four reviewers re-derived the numbers with their own code (age from the
index tag and from birth dates, the seen / never-seen split rebuilt from the training folders: same
confusion matrix 345 / 33 / 47 / 93), reviewed the experiment code on synthetic data (noise → 0.5,
planted signal found) and the labels. Nothing was found that creates the result. Scripts and raw
outputs: `~/SIAA_DATA/experiments/mil_2026-11/` (outside the repository).

## 7. A target that age does not explain: priority findings (exploratory)

**Definition** (provisional wording rules on the report conclusion, negated clauses removed): a mass
other than a microadenoma alone, a recent infarct, a haemorrhage, demyelination, true hydrocephalus,
mass effect, a vascular lesion, infection, an urgent referral or contrast advised. The experiment
below used the first version of the rules: 57 of the 558 built studies (10 %): stroke 18, tumour 15,
other 12, MS 6, haemorrhage 3, cyst 2, hydrocephalus 1. The reviewed version in the repository
(`scripts/report_priority.py`, rules-v2) differs in one study. Age alone: AUC 0.63. About 1 in 10 of
these positives is debatable on reading; the rules are for reporting, not for a gate, until
radiologists have confirmed them.

**The shipped model on this target** (never-seen, rules-v2): AUC 0.65; the flag catches 42 of 46
priority studies and is also raised on 396 of the other 472. The `prospective` stage of the release
pipeline prints this, the age baseline and the age bands for every batch from now on
(`scripts/eval_prospective.py --index … --priority …`).

**All-slice experiment.** Input rebuilt from the archived original DICOM: every axial slice of FLAIR,
T2, DWI b≈1000 and ADC; brain series chosen by frame of reference and field of view (a body station of
the same visit must not pass: the first version of the cache took abdominal DWI for brain in 23
studies); each slice on the same 240 mm field of view at 448 px; intensity per volume. 540 studies
hold all four sequences, 501 of them never-seen (370 abnormal, 43 priority). Features: a frozen
ImageNet ViT-B/16, one vector per 224 px slice. Head: trained in 5-fold patient-level
cross-validation, recipe and the two primary comparisons written down before the run.

| arm | abnormal: AUC / same-age pairs | priority: AUC / same-age pairs |
|---|---|---|
| age alone | 0.786 / 0.530 | 0.635 / 0.513 |
| shipped model, production input | 0.772 / 0.607 | 0.654 / 0.587 |
| logistic head, the 5 production slices | 0.762 / 0.577 | 0.679 / 0.614 |
| logistic head, all FLAIR slices | 0.795 / 0.619 | 0.768 / 0.757 |
| logistic head, all FLAIR + T2 slices | 0.814 / 0.641 | 0.784 / 0.773 |
| logistic head, all four sequences | 0.803 / 0.625 | 0.741 / 0.736 |
| attention over all slices, four sequences (*primary*) | 0.805 / 0.645 | 0.762 / 0.751 |
| the same design trained end to end (3-fold, last 4 blocks, *primary*) | 0.776 / 0.600 | 0.751 / 0.755 |

Primary comparisons, patient bootstrap. Abnormal: attention arm − shipped model = +0.033 (95 % CI
−0.013 … +0.077), end-to-end − shipped model = +0.005 (CI −0.037 … +0.043): **no gain shown**.
Priority: attention arm − age = +0.128 (CI +0.040 … +0.219), in same-age pairs +0.238 (CI +0.126 …
+0.343); end-to-end − age = +0.117 (CI +0.047 … +0.188), in same-age pairs +0.249 (CI +0.151 …
+0.339); end-to-end − shipped model = +0.098 (CI +0.017 … +0.180): **a real signal that is not age**,
seen with frozen features and with fine-tuning alike. It is not the scan protocol either: inside
pairs of the same protocol family the priority AUC is unchanged (0.76 – 0.79). As an operating point
(threshold read off the same scores, so descriptive only): flagging 91 % of the priority studies also
flags 59 % of the others; the shipped flag reaches 91 % by flagging 84 % of the others.

**How to read it.** Reading the whole FLAIR / T2 stack finds something the five central slices miss
and age does not explain, with only about 45 positives per training fold and generic frozen features.
An AUC of 0.76 – 0.78 is a direction, not a product. One AUC on this target has a bootstrap standard
deviation of about 0.04, so the all-slice arms cannot be ranked among themselves; DWI and ADC added
nothing here (few recent infarcts, 224 px, ImageNet features).

**What it needs.** (1) The same analysis on the next batch before believing it. (2) Priority labels
from radiologists: one click per study, in the pilot log. (3) Then a trainer and a gate for this
target, with same-age metrics and results by age band, and the abnormal / normal output kept only as a
secondary, age-adjusted grade. (4) Several hundred priority studies: the two report deliveries hold
about 110.

