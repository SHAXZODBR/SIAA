# Sentinel Medical AI — brain MRI triage for your clinic

*SIAA Medical AI, Tashkent · software release 1.1.0 · one-page overview*

## What it is

Sentinel Medical AI is decision-support software installed on one PC inside your clinic. It analyses brain MRI studies, flags the ones that may contain an abnormality so that they are read first, and drafts a report for the radiologist to edit. **A radiologist reads and signs every study.** The software has not been certified as a medical device by the Ministry of Health of the Republic of Uzbekistan or by any other regulator; it is used under the clinic's own clinical governance.

## What it does today

| Function | What it means for you |
|---|---|
| **Triage flag — brain MRI (validated)** | Tested on **178 held-out patients** from a Tashkent hospital who were not used in training: **sensitivity 0.90** (118 of 131 abnormal studies flagged) and **specificity 0.47** (22 of 47 normal studies not flagged; 95% CI 0.33–0.61). In plain words: about 1 in 10 abnormal studies is not flagged, and about half of normal studies are flagged as well. In a later check on new patients of the same hospital the flag was raised on about two of three normal studies. A flag means "read this study sooner" — it is not a diagnosis. |
| **Three languages** | Interface and report drafts in Russian, Uzbek (Latin script) and English. |
| **Offline** | Everything runs on the clinic PC without internet. No image, report or patient identifier leaves the clinic. |
| **PACS** | Studies can arrive automatically from your PACS through a local Orthanc node (optional), or are uploaded from a folder; the signed report can be filed back to the study as a DICOM document. |
| **Signed PDF** | Signing records the radiologist, the date and time and a SHA-256 fingerprint of the exact text. The PDF carries the signature block and the identity of the model that produced the AI output. |

## What it does not do

- It does not make a diagnosis and does not localise lesions.
- It never certifies a study as normal: "not flagged" is not a normal read, and the radiologist reads every study in full.
- No tumour diagnosis in this release: the public tumour classifier is switched off by default because it over-calls, and no performance is claimed for it.
- No performance is claimed for CT: a head CT is shown as "not analysed by AI" and is read as usual. Not for time-critical decisions (acute stroke or haemorrhage rule-out), children or post-operative brains.

## What a clinic needs

- One PC: Windows 10/11 64-bit, 4-core CPU (8 recommended), 8 GB RAM (16 GB recommended), 20 GB free SSD (50 GB recommended), a 1600×1000 display. No GPU and no internet connection are needed.
- Access to the studies: a PACS connection or a DICOM export folder.
- One afternoon of your IT administrator's time for the installation, and one 60-minute training session for the radiologists.

## What the pilot gives the clinic

- A prioritised worklist: flagged studies come up for reading first.
- Numbers measured on your own patients: sensitivity and specificity with 95% confidence intervals on at least 100 consecutive studies adjudicated by your radiologists (at least 40 of them normal).
- A written pilot report (agreement with your radiologists, reading time, uptime, safety events) for your own decision. Routine use only if the agreed success criteria are met.

## Contact

SIAA Medical AI · Tashkent · www.siaa.uz
Contact person: [name, position] · Phone: [+998 __ ___ __ __] · Email: [address]
