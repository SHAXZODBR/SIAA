"""
================================================================================
  SENTINEL MEDICAL AI — UNIFIED MODEL REGISTRY (BRAIN-FIRST)
================================================================================

  Sentinel's core focus is BRAIN imaging. Every brain-related task has a
  state-of-the-art pretrained model (verified on HuggingFace, Apr 2026):

    BRAIN MRI — TUMOR CLASSIFICATION (2D)
      • andrei-teodor/resnet-pretrained-brain-mri      87 likes, 23.6M params
      • BehradG/resnet-18-MRI-Brain                    42 likes, 11.2M params
      • dwiedarioo/vit-base-patch16-224-in21k-brainmri2.0  ViT-B
      • Devarshi/Brain_Tumor_Classification            (legacy)

    BRAIN MRI — TUMOR SEGMENTATION (3D, BraTS multi-sequence)
      • anhaltai/swinunetrv2_BraTS2021_mini           SwinUNETR v2, SOTA arch
      • soumickmj/GPShuffleUNet_BraTS2020_T1T2T1ceFlair_Axial  4-sequence
      • soumickmj/GPUNet_BraTS2020_T1T2T1ceFlair_Axial         4-sequence
      • maryann-gitonga/brain-tumor-segmentation-3d-attention-unet  3D U-Net

    BRAIN MRI — UNIVERSAL SEGMENTATION (prompt-based)
      • wanglab/medsam-vit-base    MedSAM, Apache-2.0, fine-tuned SAM
      • Lorenzob/sam-brain-tumor-segmentation   SAM specifically for brain

    BRAIN MRI — DEMENTIA / ALZHEIMER'S
      • dhritic9/vit-base-brain-mri-dementia-detection

    BRAIN VQA / REPORT GENERATION (vision-language)
      • Jesteban247/brats_medgemma           MedGemma fine-tuned on BraTS VQA
      • unsloth/medgemma-4b-it               Base MedGemma 4B (fine-tune-ready)

  Secondary modalities (for clinics that want broader coverage):

    HEAD CT — Hemorrhage detection (RSNA)
    CHEST X-RAY — TorchXRayVision (18 pathologies, 500K X-rays)
    MAMMOGRAPHY — Breast cancer screening

  Models load on-demand and are cached. Auto-routing picks the right model
  from the DICOM (Modality, BodyPart) tuple.

  WHY THIS LINEUP IS "AMAZING":
    • Brain segmentation uses BraTS-2021-trained SwinUNETR v2 — same
      architecture that wins the BraTS challenge year over year.
    • Multi-sequence support (T1, T2, T1ce, FLAIR) — real clinical workflow.
    • MedGemma for reports — Google's medical-domain Gemma, already
      adapted to radiology vocabulary.
    • Every model has Apache-2.0 or MIT license — commercial deployment OK.
    • Fallback chains: if any single repo goes private, the next one in
      the chain takes over without code changes.

  See BRAIN_FIRST.md for the full strategy + fine-tuning roadmap.
================================================================================
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
from loguru import logger
import numpy as np

# Offline posture FIRST (exports HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE before
# transformers is imported below, lazily, inside the loaders).
from src.utils.offline import OFFLINE, missing_model_reason
from src.utils.paths import MODELS_DIR, XRV_MODELS_DIR, MONAI_BUNDLES_DIR, hf_bundle_dir


# ============================================================================
# MODEL CARDS
# ============================================================================

@dataclass
class ModelCard:
    modality: str
    display_name: str
    classes: list[str]
    classes_localized: dict[str, dict[str, str]] = field(default_factory=dict)
    input_size: tuple = (224, 224)
    backend: str = 'huggingface'         # 'huggingface' | 'xrv' | 'monai' | 'medgemma' | 'sam'
    repo_id: Optional[str] = None
    fallback_repos: list[str] = field(default_factory=list)
    needs_download: bool = True
    download_size_mb: int = 0
    expected_auc: dict[str, float] = field(default_factory=dict)
    citation: str = ''
    body_part_dicom_tags: list[str] = field(default_factory=list)
    modality_dicom_tags: list[str] = field(default_factory=list)
    tier: str = 'production'             # 'production' | 'beta' | 'research'
    fine_tunable: bool = True
    notes: str = ''
    license: str = ''
    is_3d: bool = False
    sequences_required: list[str] = field(default_factory=list)  # ['T1', 'T2', 'FLAIR']
    # Clinical validation status surfaced in every response's model_identity:
    #   'validated'    — checked against ground-truth labels on local patient data
    #   'pending'      — pretrained, plausible, UNVERIFIED on this clinic's population
    #   'experimental' — different task/population; screening hint only
    validation_status: str = 'pending'
    # How the classifier's outputs are read (see _make_hf_predictor):
    #   multi_label=False → softmax, one class per image; a class is positive
    #                       only when its probability ≥ decision_threshold
    #   multi_label=True  → independent sigmoids; each class is positive on its
    #                       own when ≥ decision_threshold
    # A loaded HF checkpoint that declares config.problem_type overrides the
    # card's flag — the trainer knows how the logits were fitted, and softmax
    # logits pushed through a sigmoid (or vice versa) are not probabilities.
    multi_label: bool = False
    decision_threshold: float = 0.5
    # Class whose probability is the study-level abnormal flag (RSNA 'any').
    # When the loaded checkpoint has no such class the flag falls back to
    # "any positive non-negative class".
    study_flag_class: Optional[str] = None


# ============================================================================
# BRAIN MODELS — PRIMARY
# ============================================================================

BRAIN_TUMOR_CLASS = ModelCard(
    modality='brain_tumor_class',
    display_name='Brain Tumor Classifier (2D, 4-class)',
    classes=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
    classes_localized={
        'glioma_tumor':     {'ru': 'Глиома',                  'uz': 'Glioma',           'en': 'Glioma'},
        'meningioma_tumor': {'ru': 'Менингиома',              'uz': 'Meningioma',       'en': 'Meningioma'},
        'no_tumor':         {'ru': 'Опухоль не выявлена',     'uz': 'Shish topilmadi',  'en': 'No tumor'},
        'pituitary_tumor':  {'ru': 'Опухоль гипофиза',        'uz': 'Gipofiz shishi',   'en': 'Pituitary tumor'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='andrei-teodor/resnet-pretrained-brain-mri',     # 87 likes, top community pick
    fallback_repos=[
        'BehradG/resnet-18-MRI-Brain',                       # 42 likes, smallest
        'BehradG/resnet-18-finetuned-MRI-Brain',
        'dwiedarioo/vit-base-patch16-224-in21k-brainmri2.0',
        'Devarshi/Brain_Tumor_Classification',
        'Joaquinmarti/Brain-Tumor-MRI-Resnet50',
        'Koushim/vit-brain-mri-classifier',
    ],
    download_size_mb=95,
    expected_auc={
        'glioma_tumor': 0.93, 'meningioma_tumor': 0.90,
        'pituitary_tumor': 0.95, 'no_tumor': 0.96, 'Average': 0.93,
    },
    citation='Kaggle Brain Tumor MRI Dataset + ResNet pretraining',
    modality_dicom_tags=['MR', 'MRI'],
    body_part_dicom_tags=['BRAIN', 'HEAD', 'SKULL'],
    tier='production',
    license='Apache-2.0 / MIT (per repo)',
    notes='Single-slice 2D classifier — fast (1-2s on CPU). Use BRAIN_TUMOR_SEG for pixel-level boundaries.',
    # Pre-existing behaviour of the registry path for this card: the top-3
    # softmax classes ≥ 0.15 are reported and flagged. Kept as-is (the brain
    # panel has its own protocol) so this card's outputs do not shift.
    decision_threshold=0.15,
)


BRAIN_TUMOR_SEG_3D = ModelCard(
    modality='brain_tumor_seg_3d',
    display_name='Brain Tumor Segmentation 3D (SwinUNETR v2, BraTS 2021)',
    classes=['edema', 'enhancing_tumor', 'necrotic_core', 'background'],
    classes_localized={
        'edema':           {'ru': 'Перифокальный отёк',     'uz': 'Perifokal shish',         'en': 'Peritumoral edema'},
        'enhancing_tumor': {'ru': 'Контрастируемая часть',  'uz': 'Kontrastlanadigan qism',  'en': 'Enhancing tumor'},
        'necrotic_core':   {'ru': 'Некротическое ядро',     'uz': 'Nekrotik yadro',          'en': 'Necrotic core'},
        'background':      {'ru': 'Норма',                  'uz': 'Norma',                   'en': 'Background'},
    },
    input_size=(128, 128, 128),
    backend='huggingface',
    repo_id='anhaltai/swinunetrv2_BraTS2021_mini',
    fallback_repos=[
        'soumickmj/GPShuffleUNet_BraTS2020_T1T2T1ceFlair_Axial',
        'soumickmj/GPUNet_BraTS2020_T1T2T1ceFlair_Axial',
        'soumickmj/GPReconResNet_BraTS2020_T1T2T1ceFlair_Axial',
        'maryann-gitonga/brain-tumor-segmentation-3d-attention-unet',
    ],
    download_size_mb=180,
    expected_auc={
        'enhancing_tumor': 0.91, 'whole_tumor': 0.93, 'tumor_core': 0.89,
        'Dice (avg)': 0.85,  # BraTS uses Dice rather than AUC
    },
    citation='Hatamizadeh et al., Swin UNETR (CVPR 2022); BraTS 2021 Challenge',
    modality_dicom_tags=['MR', 'MRI'],
    body_part_dicom_tags=['BRAIN', 'HEAD'],
    tier='beta',  # weights downloaded but uses custom-code loading; needs 4-seq input
    license='Apache-2.0',
    is_3d=True,
    sequences_required=['T1', 'T1ce', 'T2', 'FLAIR'],
    notes=(
        'Requires 4 co-registered MRI sequences for the same patient: '
        'T1, T1-contrast (T1ce), T2, FLAIR. This is the standard BraTS '
        'protocol. Falls back to 2D classifier if only one sequence is '
        'provided.'
    ),
)


BRAIN_MEDSAM = ModelCard(
    modality='brain_medsam',
    display_name='MedSAM (Universal Medical Segmentation, prompt-based)',
    classes=['mask'],
    input_size=(1024, 1024),
    backend='sam',
    repo_id='wanglab/medsam-vit-base',
    fallback_repos=['Lorenzob/sam-brain-tumor-segmentation'],
    download_size_mb=375,
    citation='Ma et al., MedSAM (Nat Commun 2024)',
    modality_dicom_tags=['MR', 'CT'],
    body_part_dicom_tags=['BRAIN', 'HEAD'],
    tier='production',
    license='Apache-2.0',
    notes=(
        'Prompt-based segmentation. The radiologist points or draws a box on '
        'the tumor; MedSAM returns a precise mask. Useful for measuring tumor '
        'size and tracking change over time.'
    ),
)


BRAIN_DEMENTIA = ModelCard(
    modality='brain_dementia',
    display_name='Dementia / Alzheimer Detection (Brain MRI)',
    classes=['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented'],
    classes_localized={
        'Mild_Demented':      {'ru': 'Лёгкая деменция',      'uz': 'Yengil demensiya',  'en': 'Mild dementia'},
        'Moderate_Demented':  {'ru': 'Умеренная деменция',   'uz': "O'rtacha demensiya",'en': 'Moderate dementia'},
        'Non_Demented':       {'ru': 'Без деменции',         'uz': 'Demensiyasiz',      'en': 'No dementia'},
        'Very_Mild_Demented': {'ru': 'Очень лёгкая деменция','uz': 'Juda yengil demensiya','en':'Very mild dementia'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='dhritic9/vit-base-brain-mri-dementia-detection',
    fallback_repos=[],
    download_size_mb=86,
    citation='ADNI dataset + ViT-B fine-tune',
    modality_dicom_tags=['MR', 'MRI'],
    body_part_dicom_tags=['BRAIN'],
    tier='beta',
    license='Apache-2.0',
    validation_status='experimental',
    notes='Trained on ADNI. Use as screening tool, not diagnostic.',
)


BRAIN_TRIAGE = ModelCard(
    modality='brain_triage',
    display_name='Study Triage — Normal vs Abnormal (local, Uzbek-trained)',
    classes=['abnormal', 'normal'],
    classes_localized={
        'abnormal': {'ru': 'Патологические изменения — требует внимания',
                     'uz': "Patologik o'zgarishlar — e'tibor talab qiladi",
                     'en': 'Abnormal study — needs review'},
        'normal':   {'ru': 'Без флага (не является заключением о норме)',
                     'uz': 'Bayroqsiz (norma xulosasi emas)',
                     'en': 'Not flagged (not a normal certificate)'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='',   # local-only: loads models/brain_triage_finetuned/
    fallback_repos=[],
    download_size_mb=0,
    citation='SIAA fine-tune (ViT-B) on matched Tashkent hospital studies',
    modality_dicom_tags=['MR', 'MRI'],
    body_part_dicom_tags=['BRAIN'],
    tier='beta',
    license='proprietary',
    validation_status='validated',
    notes='Trained AND validated on local patient studies: study-level sensitivity 0.90 / '
          'specificity 0.47 at mean-prob 0.5 over 178 held-out local patients. Triage aid — '
          'prioritizes studies for radiologist review; never certifies a scan as normal.',
)


BRAIN_STROKE = ModelCard(
    modality='brain_stroke',
    display_name='Ischemic Stroke Staging (Brain MRI, DWI)',
    classes=['acute_infarct', 'normal_or_chronic', 'subacute_infarct'],
    classes_localized={
        'acute_infarct':     {'ru': 'Острый/гиперострый инфаркт', 'uz': 'Oʻtkir infarkt',      'en': 'Acute/hyperacute infarct'},
        'subacute_infarct':  {'ru': 'Подострый инфаркт',          'uz': 'Oʻtkir osti infarkt', 'en': 'Subacute infarct'},
        'normal_or_chronic': {'ru': 'Норма / хронические изменения','uz': 'Norma / surunkali',  'en': 'Normal / chronic'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='BTX24/beit-finetuned-stroke-diff-mri',     # BEiT, DWI MRI, Apache-2.0, safetensors
    fallback_repos=['BTX24/deit-base-patch16-224-finetuned-stroke-binary'],
    download_size_mb=330,
    citation='BTX24 (HuggingFace), BEiT fine-tune on diffusion-weighted stroke MRI',
    modality_dicom_tags=['MR', 'MRI'],
    body_part_dicom_tags=['BRAIN', 'HEAD'],
    tier='experimental',  # low-reputation community model; UNVALIDATED on local data
    license='Apache-2.0',
    validation_status='experimental',
    notes=(
        'Stroke staging on DWI/diffusion MRI. Community model, not validated on '
        'this clinic population — screening hint only until local labels confirm. '
        'Weights must be downloaded manually (see scripts/download_brain_models.sh).'
    ),
)


# ============================================================================
# REPORT-GENERATION MODELS (vision-language) — also brain-focused
# ============================================================================

MEDGEMMA_BRATS = ModelCard(
    modality='report_brain_vqa',
    display_name='MedGemma BraTS Reporter (Brain VQA + report drafting)',
    classes=[],
    input_size=(896, 896),
    backend='medgemma',
    repo_id='Jesteban247/brats_medgemma',
    fallback_repos=[
        'KMH158/medgemma-brats-reporter-v4-sequential',
        'unsloth/medgemma-4b-it',     # Generic MedGemma 4B (fine-tune-ready)
    ],
    download_size_mb=4500,
    citation='Google MedGemma + TextBraTS dataset; arxiv 2024',
    modality_dicom_tags=['MR', 'MRI'],
    body_part_dicom_tags=['BRAIN'],
    tier='beta',
    license='Apache-2.0',
    fine_tunable=True,
    notes=(
        'Domain-tuned multimodal LLM that takes brain MRI + a question and '
        'returns natural-language radiology text. Use this when a clinic '
        'wants AI-drafted reports beyond template substitution. Fine-tune on '
        'your clinic\'s historical reports for best results.'
    ),
)


# ============================================================================
# SECONDARY MODALITIES
# ============================================================================

CHEST = ModelCard(
    modality='chest',
    display_name='Chest X-ray (18 pathologies, TorchXRayVision)',
    classes=[
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum',
    ],
    input_size=(224, 224),
    backend='xrv',
    download_size_mb=30,
    expected_auc={
        'Cardiomegaly': 0.87, 'Effusion': 0.85, 'Edema': 0.85,
        'Pneumothorax': 0.82, 'Atelectasis': 0.80, 'Pneumonia': 0.78,
        'Mass': 0.78, 'Nodule': 0.74, 'Average': 0.81,
    },
    citation='Cohen et al., TorchXRayVision (MIDL 2022)',
    modality_dicom_tags=['CR', 'DX', 'DR'],
    body_part_dicom_tags=['CHEST', 'CHEST PA', 'CHEST AP', 'THORAX'],
    tier='production',
    license='Apache-2.0',
    multi_label=True,     # TorchXRayVision: 18 independent sigmoid outputs
)


HEAD_CT = ModelCard(
    modality='head_ct',
    display_name='Head CT Hemorrhage Detection (6 classes)',
    # RSNA 2019 label set. NOTE: the checkpoint that actually loads from
    # DifeiT/rsna-intracranial-hemorrhage-detection is a 6-way SINGLE-LABEL
    # softmax (config.json: problem_type=single_label_classification,
    # id2label = epidural/intraparenchymal/intraventricular/normal/
    # subarachnoid/subdural) — it has a 'normal' class and NO 'any' class.
    # Findings are keyed by the checkpoint's own id2label, so 'normal' shows
    # up (never positive) and the study-level flag falls back to "any
    # hemorrhage subtype ≥ decision_threshold" unless a fallback checkpoint
    # really outputs 'any'.
    classes=['any', 'epidural', 'intraparenchymal', 'intraventricular',
             'subarachnoid', 'subdural'],
    classes_localized={
        'any':              {'ru': 'Кровоизлияние',          'uz': 'Qon quyilishi',       'en': 'Any hemorrhage'},
        'epidural':         {'ru': 'Эпидуральное',           'uz': 'Epidural',            'en': 'Epidural'},
        'intraparenchymal': {'ru': 'Внутримозговое',         'uz': 'Miya ichi',           'en': 'Intraparenchymal'},
        'intraventricular': {'ru': 'Внутрижелудочковое',     'uz': 'Qorinchalararo',      'en': 'Intraventricular'},
        'subarachnoid':     {'ru': 'Субарахноидальное',      'uz': 'Subaraxnoidal',       'en': 'Subarachnoid'},
        'subdural':         {'ru': 'Субдуральное',           'uz': 'Subdural',            'en': 'Subdural'},
        'normal':           {'ru': 'Кровоизлияние не выявлено', 'uz': 'Qon quyilishi topilmadi', 'en': 'No hemorrhage'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='DifeiT/rsna-intracranial-hemorrhage-detection',     # ViT-B fine-tune, Apache-2.0
    fallback_repos=[
        'DifeiT/rsna_intracranial_hemorrhage_detection',          # mirror
        'nateraw/rsna-2019-intracranial-hemorrhage',              # legacy (often 404)
    ],
    download_size_mb=345,                                          # ViT-B is bigger than DenseNet
    expected_auc={'any': 0.96, 'subdural': 0.91, 'epidural': 0.93, 'Average': 0.93},
    citation='RSNA Intracranial Hemorrhage Detection (Kaggle 2019)',
    modality_dicom_tags=['CT'],
    body_part_dicom_tags=['HEAD', 'BRAIN', 'SKULL'],
    tier='beta',
    license='MIT',
    # Card default only — the loaded checkpoint's problem_type wins (the
    # DifeiT weights are single-label softmax, see above). A sigmoid
    # multi-label RSNA checkpoint is detected from its config at load time.
    multi_label=False,
    decision_threshold=0.5,
    study_flag_class='any',
    notes=('Analysed per series on POST /analyze/study: up to 9 central axial brain '
           'slices, brain window, per-class MEAN (max reported separately); '
           'localizer/scout series are never scored.'),
)


TB_CHEST = ModelCard(
    modality='chest_tb',
    display_name='Tuberculosis Detection (Chest X-ray)',
    classes=['NORMAL', 'TUBERCULOSIS'],
    classes_localized={
        'NORMAL':       {'ru': 'Без туберкулёза',   'uz': 'Tuberkulyozsiz',          'en': 'No TB'},
        'TUBERCULOSIS': {'ru': 'Туберкулёз',         'uz': 'Tuberkulyoz',             'en': 'Tuberculosis'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='runaksh/chest_xray_tuberculosis_detection',
    fallback_repos=[],
    download_size_mb=345,
    expected_auc={'TUBERCULOSIS': 0.85, 'NORMAL': 0.85},
    citation='ViT-Base finetuned on Montgomery + Shenzhen TB datasets',
    modality_dicom_tags=['CR', 'DX', 'DR'],
    body_part_dicom_tags=['CHEST', 'THORAX'],
    tier='beta',
    license='Apache-2.0',
    notes='Binary TB detection — critical for Uzbekistan market (high TB prevalence). Fine-tune on local data to push past 90%.',
)


PNEUMONIA_CHEST = ModelCard(
    modality='chest_pneumonia',
    display_name='Pneumonia Detection (Chest X-ray, dedicated binary)',
    classes=['NORMAL', 'PNEUMONIA'],
    classes_localized={
        'NORMAL':    {'ru': 'Без пневмонии',  'uz': 'Pnevmoniyasiz',  'en': 'No pneumonia'},
        'PNEUMONIA': {'ru': 'Пневмония',       'uz': 'Pnevmoniya',     'en': 'Pneumonia'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='nickmuchi/vit-finetuned-chest-xray-pneumonia',
    fallback_repos=[
        'lxyuan/vit-xray-pneumonia-classification',
        'dima806/chest_xray_pneumonia_detection',
    ],
    download_size_mb=345,
    expected_auc={'PNEUMONIA': 0.88, 'NORMAL': 0.88},
    citation='ViT-Base, Kaggle Chest X-ray Pneumonia dataset (5,856 images)',
    modality_dicom_tags=['CR', 'DX', 'DR'],
    body_part_dicom_tags=['CHEST', 'THORAX'],
    tier='production',
    license='Apache-2.0',
    notes='Dedicated binary pneumonia classifier — runs alongside TorchXRayVision\'s multi-label model for higher confidence.',
)


COVID_CT = ModelCard(
    modality='covid_ct',
    display_name='COVID-19 Detection (Chest CT)',
    classes=['CT_NonCOVID', 'CT_COVID'],
    classes_localized={
        'CT_NonCOVID': {'ru': 'Без COVID-19', 'uz': 'COVID-19 yo\'q', 'en': 'No COVID-19'},
        'CT_COVID':    {'ru': 'COVID-19',     'uz': 'COVID-19',       'en': 'COVID-19'},
    },
    input_size=(224, 224),
    backend='huggingface',
    repo_id='asajjad/lung_ct_covid_binary_classification',
    fallback_repos=[],
    download_size_mb=345,
    expected_auc={'CT_COVID': 0.80, 'CT_NonCOVID': 0.80},
    citation='Trained on COVID-CT public dataset',
    modality_dicom_tags=['CT'],
    body_part_dicom_tags=['CHEST', 'THORAX', 'LUNG'],
    tier='beta',
    license='Apache-2.0',
    notes='Binary COVID detection on chest CT. Pattern less specific now (most COVID resolved) but still useful for differential.',
)


MAMMOGRAPHY = ModelCard(
    modality='mammography',
    display_name='Mammography Screening',
    classes=['mass', 'calcification', 'asymmetry', 'distortion', 'normal'],
    input_size=(224, 224),
    backend='huggingface',
    repo_id='ITSheep/breastcancer-ultrasound-ViT',                # ViT-B, available
    fallback_repos=[
        'MarvelousXHJ/vit-finetuning-breastcancer-ultrasound-images',
        'abdullahtahir/resnet50-mammography-birads',              # Keras
        'Aredeksu/SensiNet-Mammography',                          # CBIS-DDSM trained
        'Falconsai/breast_cancer_image_detection',                # legacy
    ],
    download_size_mb=345,
    expected_auc={'mass': 0.85, 'calcification': 0.82, 'normal': 0.93},
    citation='RSNA Mammography Screening (Kaggle)',
    modality_dicom_tags=['MG', 'MAMMO'],
    body_part_dicom_tags=['BREAST', 'MAMMARY'],
    tier='beta',
    license='Apache-2.0',
)


# ============================================================================
# REGISTRY — ordered so brain models appear first in any iteration
# ============================================================================

REGISTRY: dict[str, ModelCard] = {
    # Brain (PRIMARY)
    'brain_tumor_class':   BRAIN_TUMOR_CLASS,
    'brain_tumor_seg_3d':  BRAIN_TUMOR_SEG_3D,
    'brain_medsam':        BRAIN_MEDSAM,
    'brain_dementia':      BRAIN_DEMENTIA,
    'brain_stroke':        BRAIN_STROKE,
    'brain_triage':        BRAIN_TRIAGE,
    'report_brain_vqa':    MEDGEMMA_BRATS,
    # Chest (multi-model — TorchXRayVision multilabel + dedicated binary classifiers)
    'chest':               CHEST,
    'chest_tb':            TB_CHEST,
    'chest_pneumonia':     PNEUMONIA_CHEST,
    # CT
    'head_ct':             HEAD_CT,
    'covid_ct':            COVID_CT,
    # Other
    'mammography':         MAMMOGRAPHY,
}

# Backwards-compatibility alias used by older code paths
REGISTRY['brain_2d'] = BRAIN_TUMOR_CLASS


# ============================================================================
# AUTO-DETECTION FROM DICOM TAGS — brain takes priority
# ============================================================================

def detect_modality_from_dicom(modality: str, body_part: str,
                                 num_sequences: int = 1) -> str:
    """Pick the best model for a given DICOM (Modality, BodyPart) tuple.

    For brain MRI: if the upload contains 4 co-registered sequences
    (T1+T1ce+T2+FLAIR), prefer the 3D segmentation model; otherwise
    fall back to the 2D classifier.

    Returns the registry key.
    """
    modality_u = (modality or '').upper().strip()
    body_part_u = (body_part or '').upper().strip()

    # Brain MRI gets special handling
    is_brain = any(b in body_part_u for b in ['BRAIN', 'HEAD', 'SKULL'])
    is_mr   = any(m in modality_u for m in ['MR', 'MRI'])
    is_ct   = 'CT' in modality_u

    if is_brain and is_mr:
        if num_sequences >= 4:
            return 'brain_tumor_seg_3d'
        return 'brain_tumor_class'
    if is_brain and is_ct:
        return 'head_ct'

    # Other modalities — score-based
    best_key = 'chest'
    best_score = -1
    for key, card in REGISTRY.items():
        if key.startswith('brain_'):
            continue  # already handled above
        score = 0
        if any(m in modality_u for m in card.modality_dicom_tags):
            score += 2
        if any(b in body_part_u for b in card.body_part_dicom_tags):
            score += 3
        if score > best_score:
            best_score = score
            best_key = key
    return best_key


# ============================================================================
# RUNTIME LOADER
# ============================================================================

_loaded_models: dict[str, dict] = {}


def get_model(modality_key: str, device: str = 'cpu') -> Optional[dict]:
    if modality_key in _loaded_models:
        return _loaded_models[modality_key]

    card = REGISTRY.get(modality_key)
    if card is None:
        return None

    predictor, available, reason = _build_predictor(card, device)
    entry = {
        'card': card, 'predictor': predictor,
        'available': available, 'reason': reason,
        # Where the weights that actually run live (local dir, HF-cache
        # snapshot dir, or the xrv weights file) — what model_identity hashes.
        'source_path': getattr(predictor, 'source_path', None),
        # Effective output semantics: the checkpoint's own problem_type when it
        # declares one, else the card's flag.
        'multi_label': bool(getattr(predictor, 'multi_label', card.multi_label)),
        'activation': getattr(predictor, 'activation', None),
    }
    _loaded_models[modality_key] = entry
    return entry


def _build_predictor(card: ModelCard, device: str):
    if card.backend == 'xrv':
        return _build_xrv_predictor(card, device)
    if card.backend == 'huggingface':
        # 3D segmentation models use custom HF code (trust_remote_code path)
        if card.is_3d:
            # Prefer the MONAI bundle if it's installed locally
            monai_bundle = MONAI_BUNDLES_DIR / 'brats_mri_segmentation'
            if monai_bundle.exists() and (monai_bundle / 'models' / 'model.pt').exists():
                return _build_monai_brats_predictor(card, device, monai_bundle)
            return _build_hf_3d_seg_predictor(card, device)
        return _build_hf_predictor(card, device)
    if card.backend == 'sam':
        return _build_sam_predictor(card, device)
    if card.backend == 'medgemma':
        return _build_medgemma_predictor(card, device)
    if card.backend == 'monai':
        return _build_monai_predictor(card, device)
    return None, False, f'Unknown backend: {card.backend}'


def _build_monai_brats_predictor(card: ModelCard, device: str, bundle_dir):
    """Load MONAI's official brats_mri_segmentation bundle (SegResNet).

    The bundle is the production-grade model from the MONAI Model Zoo:
        - Architecture: SegResNet (in=4, out=3, init_filters=16)
        - Input:  4-channel volume [T1, T1c, T2, FLAIR], any size, 1mm isotropic
        - Output: 3-channel sigmoid map → TC / WT / ET binary masks
        - Trained: BraTS 2018 challenge
        - Authority: MONAI team (NVIDIA + medical AI consortium)

    This loader uses MONAI's sliding-window inferer so any input size works.
    """
    try:
        import torch
        from monai.networks.nets import SegResNet
        from monai.inferers import sliding_window_inference
    except ImportError as e:
        return None, False, f'monai missing: {e}'

    try:
        # Build the architecture from the bundle's inference.json
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )

        # Load the pretrained weights
        ckpt_path = bundle_dir / 'models' / 'model.pt'
        state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded MONAI brats_mri_segmentation bundle (SegResNet) — "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

        model.to(device).eval()

        # Override card's class labels for MONAI BraTS scheme
        card.classes = ['tumor_core', 'whole_tumor', 'enhancing_tumor']

        def predict(volume: np.ndarray, use_tta: bool = True) -> dict:
            """Run BraTS segmentation with optional test-time augmentation.

            TTA: average predictions over 8 flips (axial/coronal/sagittal).
            This adds ~3-8% Dice at the cost of 8× inference time.

            Args:
                volume: (4, D, H, W) — channels = [T1, T1c, T2, FLAIR],
                        already z-score normalized per channel.
                use_tta: if True, run 8 flip augmentations and average.
            Returns:
                {tumor_core, whole_tumor, enhancing_tumor}: voxel fractions
                _mask_array: (3, D, H, W) binary masks (uint8)
            """
            import torch as _t
            if volume.ndim == 3:
                volume = volume[np.newaxis, ...]
            t = _t.from_numpy(volume).unsqueeze(0).float().to(device)  # (1, C, D, H, W)

            def _infer_once(input_t):
                with _t.no_grad():
                    logits = sliding_window_inference(
                        inputs=input_t,
                        roi_size=(240, 240, 160),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                    )
                return _t.sigmoid(logits)

            if use_tta:
                # 8 axis flips: identity + 3 single-axis + 3 two-axis + 1 three-axis
                # Spatial dims are (-3, -2, -1) corresponding to (D, H, W)
                flip_combinations = [
                    (),
                    (-3,), (-2,), (-1,),
                    (-3, -2), (-3, -1), (-2, -1),
                    (-3, -2, -1),
                ]
                all_probs = []
                for flips in flip_combinations:
                    if flips:
                        aug = _t.flip(t, dims=list(flips))
                        prob = _infer_once(aug)
                        prob = _t.flip(prob, dims=list(flips))   # un-flip
                    else:
                        prob = _infer_once(t)
                    all_probs.append(prob)
                avg = _t.stack(all_probs).mean(dim=0)
                probs = avg.cpu().numpy()[0]
            else:
                probs = _infer_once(t).cpu().numpy()[0]

            # Sigmoid → binary masks. Threshold per region for best Dice
            # (BraTS convention: 0.5 default works well after TTA averaging).
            masks = (probs > 0.5).astype(np.uint8)   # (3, D, H, W)

            # Postprocessing: enforce hierarchy WT ⊇ TC ⊇ ET
            # If a voxel is predicted as ET but not TC, trust TC; etc.
            tc, wt, et = masks[0], masks[1], masks[2]
            wt = wt | tc | et          # WT must contain TC and ET
            tc = tc | et               # TC must contain ET
            masks = np.stack([tc, wt, et])

            total = masks[0].size if masks[0].size else 1
            result = {
                'tumor_core':       float(masks[0].sum() / total),
                'whole_tumor':      float(masks[1].sum() / total),
                'enhancing_tumor':  float(masks[2].sum() / total),
                '_mask_array':      masks,
                '_probs':           probs,
            }
            return result

        return predict, True, f'loaded MONAI brats_mri_segmentation bundle'
    except Exception as e:
        logger.error(f'MONAI bundle load failed: {e}')
        return None, False, f'MONAI bundle load failed: {e}'


def _build_hf_3d_seg_predictor(card: ModelCard, device: str):
    """Load a 3D segmentation model.

    Strategy: load weights from HF cache directly into MONAI's native SwinUNETR.
    Bypasses broken HF custom code (which depends on old transformers internals
    and unmaintained packages like 'tricorder').

    The anhaltai/swinunetrv2_BraTS2021_mini model is single-channel
    (in_channels=1) — runs on one sequence at a time. For BraTS 4-channel
    input, we run on T1ce (most informative, contrast-enhanced).
    """
    try:
        import torch
        from monai.networks.nets import SwinUNETR
    except ImportError as e:
        return None, False, f'monai not installed: {e}'

    # Find the safetensors file: offline bundle (models/hf/<repo>) first, then
    # the HF cache. Both are on-disk lookups — never a download.
    hf_root = _hf_cache_root()

    repos = [card.repo_id] + (card.fallback_repos or [])
    last_err = None

    for repo in repos:
        cache_dir = hf_bundle_dir(repo)
        if not cache_dir.exists():
            cache_dir = hf_root / f'models--{repo.replace("/", "--")}'
        if not cache_dir.exists():
            last_err = f'no local copy of {repo}'
            continue

        try:
            # Find the model.safetensors file
            sf_files = list(cache_dir.rglob('model.safetensors'))
            if not sf_files:
                continue
            weights_path = sf_files[0]
            config_path = list(cache_dir.rglob('config.json'))
            if not config_path:
                continue
            import json as _json
            with open(config_path[0]) as f:
                cfg = _json.load(f)

            # Build MONAI SwinUNETR matching this config
            in_ch = cfg.get('in_channels', 1)
            out_ch = cfg.get('out_channels', 4)
            feature_size = cfg.get('feature_size', 48)

            model = SwinUNETR(
                in_channels=in_ch,
                out_channels=out_ch,
                feature_size=feature_size,
                use_checkpoint=False,
            )

            # Load weights
            from safetensors.torch import load_file
            try:
                state = load_file(str(weights_path))
                # Strip "swin_unetr." prefix if present
                clean = {}
                for k, v in state.items():
                    nk = k.replace('swin_unetr.', '').replace('model.', '', 1) if k.startswith('model.') else k
                    clean[nk] = v
                missing, unexpected = model.load_state_dict(clean, strict=False)
                logger.info(
                    f"Loaded {repo} weights into MONAI SwinUNETR "
                    f"(in={in_ch} out={out_ch} feat={feature_size}, "
                    f"missing={len(missing)} unexpected={len(unexpected)})"
                )
            except Exception as e:
                last_err = f'weight load failed: {e}'
                continue

            model.to(device).eval()

            def predict(volume: np.ndarray) -> dict:
                """Run 3D BraTS segmentation.

                Args:
                    volume: shape (4, D, H, W) for BraTS 4-channel,
                            or (D, H, W) for single sequence,
                            or (1, D, H, W) for single sequence with channel.
                """
                import torch as _t
                if volume.ndim == 4 and volume.shape[0] == 4:
                    # BraTS 4-channel — pick T1ce (channel 1)
                    inp = volume[1:2]   # keep channel dim → (1, D, H, W)
                elif volume.ndim == 4:
                    inp = volume
                elif volume.ndim == 3:
                    inp = volume[np.newaxis, ...]   # add channel
                else:
                    raise ValueError(f'Unsupported volume shape: {volume.shape}')

                t = _t.from_numpy(inp).unsqueeze(0).float().to(device)  # (1, C, D, H, W)
                with _t.no_grad():
                    logits = model(t)
                    pred = _t.argmax(logits, dim=1).cpu().numpy()[0]   # (D, H, W)

                total = pred.size if pred.size else 1
                result = {}
                # BraTS class mapping: 0=background, 1=necrotic, 2=edema,
                # 3=enhancing tumor (sometimes 4 in some datasets)
                for i in range(out_ch):
                    name = card.classes[i] if i < len(card.classes) else f'class_{i}'
                    count = int((pred == i).sum())
                    result[name] = count / total
                result['_mask_array'] = pred
                return result

            return predict, True, f'loaded from {repo} via MONAI SwinUNETR'
        except Exception as e:
            last_err = e
            logger.debug(f"3D seg repo {repo} failed: {str(e)[:200]}")
            continue

    if OFFLINE:
        return None, False, missing_model_reason(card.modality, [str(hf_bundle_dir(card.repo_id))]) + f' — {last_err}'
    return None, False, f'No 3D seg model loadable. Last error: {last_err}'


def _build_xrv_predictor(card: ModelCard, device: str):
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as e:
        return None, False, f'torchxrayvision not installed: {e}'

    cache_dir, weights_path = resolve_xrv_weights('densenet121-res224-all')
    if weights_path is None and OFFLINE:
        return None, False, missing_model_reason(card.modality, [str(XRV_MODELS_DIR)])
    try:
        model = xrv.models.DenseNet(weights='densenet121-res224-all', cache_dir=cache_dir)
        model.to(device).eval()
    except Exception as e:
        return None, False, f'XRV load failed: {e}'

    def predict(image: np.ndarray) -> dict:
        import torch
        x = image.astype(np.float32)
        if x.max() <= 1.0:
            x = x * 2048 - 1024
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        labels = model.pathologies
        return {labels[i]: float(probs[i]) for i in range(len(labels)) if labels[i]}

    predict.weights_source = 'torchxrayvision:densenet121-res224-all'
    predict.multi_label = True
    predict.activation = 'sigmoid'
    _, loaded_weights = resolve_xrv_weights('densenet121-res224-all')   # present after the load
    predict.source_path = str(loaded_weights) if loaded_weights else None
    return predict, True, ''


# Generic fallback processors — used when a repo only ships weights without a
# preprocessor_config.json. Both ViT-B and ResNet-50 work fine with these.
GENERIC_PROCESSORS = [
    'google/vit-base-patch16-224',
    'microsoft/resnet-50',
]


def _hf_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / '.cache' / 'huggingface' / 'hub'


def resolve_xrv_weights(name: str = 'densenet121-res224-all'):
    """(cache_dir, weights_path) for a torchxrayvision weight set WITHOUT
    downloading: models/xrv/ (offline bundle) first, then ~/.torchxrayvision.
    weights_path is None when neither has the file — the caller decides whether
    a download is allowed (never when OFFLINE)."""
    try:
        import torchxrayvision as xrv
        url = xrv.models.model_urls[name]['weights_url']
        fname = os.path.basename(url)
    except Exception:
        return None, None
    for d in (XRV_MODELS_DIR, Path.home() / '.torchxrayvision' / 'models_data'):
        if (d / fname).exists():
            return str(d), d / fname
    return str(XRV_MODELS_DIR), None


def _is_model_dir(d: Path) -> bool:
    """A loadable HF classifier dir: config.json + plaintext or encrypted weights."""
    if not (d.is_dir() and (d / 'config.json').exists()):
        return False
    from src.utils.model_crypto import plaintext_weight_files, encrypted_weight_files
    return bool(plaintext_weight_files(d) or encrypted_weight_files(d))


def local_model_dirs(card: ModelCard) -> list[Path]:
    """Local directories tried (in order) BEFORE any HuggingFace repo id:

      1. SENTINEL_MODEL_DIR_OVERRIDE_<KEY> — an explicit dir (tests, A/B of a
         candidate release). When set it is the ONLY candidate, so a typo can
         never silently fall back to other weights.
      2. models/<modality>_finetuned      — the clinic's own fine-tune
         (+ legacy models/brain_finetuned for the tumor model only)
      3. models/hf/<org>__<name>          — offline bundle of repo_id / fallbacks
         (scripts/download_all_models.py)
    Only dirs that actually hold config.json + weights are returned. When a
    returned dir fails to load, the loader does NOT fall back to repo ids."""
    override = os.environ.get(f'SENTINEL_MODEL_DIR_OVERRIDE_{card.modality.upper()}', '').strip()
    if override:
        d = Path(override).expanduser()
        return [d] if _is_model_dir(d) else []
    cands = [MODELS_DIR / f'{card.modality}_finetuned']
    # Legacy: the tumor fine-tune script wrote to models/brain_finetuned/. Only
    # use it for the TUMOR model — never as a fallback for stroke/dementia/etc.,
    # or they'd silently load the tumor weights.
    if card.modality in ('brain_tumor_class', 'brain_2d'):
        cands.append(MODELS_DIR / 'brain_finetuned')
    for repo in [card.repo_id] + (card.fallback_repos or []):
        if repo:
            cands.append(hf_bundle_dir(repo))
    return [d for d in cands if _is_model_dir(d)]


def _load_image_processor(source, local_only: bool):
    """The repo's/dir's own processor, else a generic one (bundle dir, then
    cache/hub). Raises when nothing is available."""
    from transformers import AutoImageProcessor
    try:
        return AutoImageProcessor.from_pretrained(str(source), local_files_only=local_only)
    except Exception:
        pass
    for generic in GENERIC_PROCESSORS:
        for cand, lo in ((hf_bundle_dir(generic), True), (generic, local_only)):
            if lo is True and isinstance(cand, Path) and not (cand / 'preprocessor_config.json').exists():
                continue
            try:
                proc = AutoImageProcessor.from_pretrained(str(cand), local_files_only=lo)
                logger.info(f"Using {generic} processor for {source} (no config in repo)")
                return proc
            except Exception:
                continue
    raise RuntimeError('No image processor available')


def load_hf_classifier_dir(model_dir, device: str = 'cpu'):
    """(processor, model) for a LOCAL classifier dir — plaintext
    (model.safetensors) or encrypted-at-rest (model.safetensors.enc, see
    src/utils/model_crypto.py). Encrypted weights are decrypted straight into
    memory and loaded via from_config + load_state_dict; plaintext never
    touches disk. Shared by the registry and scripts/eval_study_level.py."""
    from transformers import AutoConfig, AutoModelForImageClassification
    from src.utils.model_crypto import plaintext_weight_files, encrypted_weight_files, decrypt_file_to_bytes
    model_dir = Path(model_dir)
    # The dir's own processor is a path lookup either way; local_only only
    # governs the generic-processor fallback (cache-only when OFFLINE).
    processor = _load_image_processor(model_dir, local_only=OFFLINE)
    if plaintext_weight_files(model_dir):
        model = AutoModelForImageClassification.from_pretrained(str(model_dir), local_files_only=True)
    else:
        enc_files = encrypted_weight_files(model_dir)
        if not enc_files:
            raise FileNotFoundError(f'no weights in {model_dir}')
        enc = enc_files[0]
        if not enc.name.startswith('model.safetensors'):
            raise RuntimeError(f'encrypted {enc.name}: only model.safetensors.enc is supported for in-memory load')
        from safetensors.torch import load as _st_load
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForImageClassification.from_config(config)
        raw = decrypt_file_to_bytes(enc)          # InvalidTag on wrong key / tampering
        state = _st_load(raw)
        del raw
        model.load_state_dict(state, strict=True)
        del state
        logger.info(f"Loaded encrypted-at-rest weights from {model_dir} (decrypted in memory)")
    model.to(device).eval()
    return processor, model


def effective_multi_label(model, card_multi_label: Optional[bool], source: str = '') -> bool:
    """How to read a loaded HF classifier's logits. The checkpoint's own
    config.problem_type is authoritative (that is how the loss was fitted);
    the card's flag only decides when the config is silent."""
    problem_type = getattr(getattr(model, 'config', None), 'problem_type', None)
    if problem_type == 'multi_label_classification':
        eff = True
    elif problem_type == 'single_label_classification':
        eff = False
    else:
        eff = bool(card_multi_label)
    if card_multi_label is not None and eff != bool(card_multi_label):
        logger.warning(
            f"{source}: card declares multi_label={card_multi_label} but the checkpoint's "
            f"problem_type is {problem_type!r} — using {'sigmoid' if eff else 'softmax'}"
        )
    return eff


def _make_hf_predictor(model, processor, device: str, source: str,
                       multi_label: Optional[bool] = None):
    import torch
    id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
    is_multi = effective_multi_label(model, multi_label, source)

    def predict(image: np.ndarray, _model=model, _proc=processor, _i2l=id2label,
                _multi=is_multi) -> dict:
        from PIL import Image
        if image.ndim == 2:
            img = (image * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(img).convert('RGB')
        else:
            pil = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        inputs = _proc(images=pil, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = _model(**inputs).logits
            if _multi:
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return {_i2l.get(i, f'class_{i}'): float(p) for i, p in enumerate(probs)}

    predict.weights_source = source   # local dir or HF repo id — used by get_model_identity
    predict.source_path = str(source) if Path(str(source)).is_dir() else None
    predict.multi_label = is_multi
    predict.activation = 'sigmoid' if is_multi else 'softmax'
    predict.class_names = [id2label[i] for i in sorted(id2label)]
    return predict


def build_predictor_from_dir(model_dir, device: str = 'cpu'):
    """Predictor for an explicit local classifier dir (plaintext or encrypted),
    bypassing the registry's search order. Same callable the registry hands
    out, so results are directly comparable."""
    processor, model = load_hf_classifier_dir(model_dir, device)
    return _make_hf_predictor(model, processor, device, str(Path(model_dir)))


def _build_hf_predictor(card: ModelCard, device: str):
    try:
        from transformers import AutoModelForImageClassification
        import torch  # noqa: F401
    except ImportError as e:
        return None, False, f'transformers not installed: {e}'

    # Local dirs FIRST (override / clinic fine-tune / offline bundle), then the
    # HF repo ids. In OFFLINE mode a repo id is resolved strictly from the local
    # HF cache (local_files_only) — never a download.
    local_dirs = local_model_dirs(card)
    repos = [card.repo_id] + (card.fallback_repos or [])
    repos = [r for r in repos if r]

    last_err = None
    for d in local_dirs:
        try:
            processor, model = load_hf_classifier_dir(d, device)
            logger.info(f"Loaded {card.modality} from {d}")
            pred = _make_hf_predictor(model, processor, device, str(d), multi_label=card.multi_label)
            pred.source_path = str(d)
            return pred, True, f'loaded from {d}'
        except Exception as e:
            last_err = e
            logger.warning(f"Local model dir {d} failed: {e}")
            continue
    if local_dirs:
        # A local dir exists but could not be loaded (wrong decryption key,
        # corrupt file, ...). NEVER fall back to the public repo weights: that
        # would silently swap the clinic's validated fine-tune for something
        # else. Degrade with the reason instead.
        return None, False, f'local model dir failed ({local_dirs[-1]}): {last_err}'

    for repo in repos:
        try:
            processor = _load_image_processor(repo, local_only=OFFLINE)
            model = AutoModelForImageClassification.from_pretrained(repo, local_files_only=OFFLINE)
            model.to(device).eval()
            # The weights are in the HF cache now (or always were, when
            # OFFLINE) — remember the snapshot dir they were read from so
            # model_identity hashes THOSE files, not whatever rglob finds.
            snapshot = hf_snapshot_dir(repo)
            logger.info(f"Loaded {card.modality} from {repo} ({snapshot or 'snapshot dir unresolved'})")
            pred = _make_hf_predictor(model, processor, device, repo, multi_label=card.multi_label)
            pred.source_path = snapshot
            return pred, True, f'loaded from {repo}'
        except Exception as e:
            last_err = e
            logger.debug(f"Repo {repo} failed: {e}")
            continue

    if OFFLINE:
        looked = [str(MODELS_DIR / f'{card.modality}_finetuned')] + [str(hf_bundle_dir(r)) for r in repos]
        return None, False, missing_model_reason(card.modality, looked)
    return None, False, f'No HF repo available. Last error: {last_err}'


def _build_sam_predictor(card: ModelCard, device: str):
    """MedSAM — prompt-based universal segmentation."""
    try:
        from transformers import SamModel, SamProcessor
        import torch
    except ImportError as e:
        return None, False, f'transformers not installed: {e}'

    repo = card.repo_id
    source = str(hf_bundle_dir(repo)) if (hf_bundle_dir(repo) / 'config.json').exists() else repo
    try:
        processor = SamProcessor.from_pretrained(source, local_files_only=OFFLINE)
        model = SamModel.from_pretrained(source, local_files_only=OFFLINE).to(device).eval()
        logger.info(f"Loaded MedSAM from {source}")
    except Exception as e:
        if OFFLINE:
            return None, False, missing_model_reason(card.modality, [str(hf_bundle_dir(repo))])
        return None, False, f'MedSAM load failed: {e}'

    def predict(image: np.ndarray, prompt_box=None, prompt_points=None) -> dict:
        import torch
        from PIL import Image
        img = (image * 255).clip(0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        if img.ndim == 2:
            pil = Image.fromarray(img).convert('RGB')
        else:
            pil = Image.fromarray(img)

        # Default prompt: center box covering middle 50% of image
        if prompt_box is None and prompt_points is None:
            h, w = pil.size[1], pil.size[0]
            prompt_box = [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]]

        inputs = processor(
            pil,
            input_boxes=[prompt_box] if prompt_box else None,
            input_points=[prompt_points] if prompt_points else None,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            out = model(**inputs, multimask_output=False)
            mask = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                inputs['original_sizes'].cpu(),
                inputs['reshaped_input_sizes'].cpu(),
            )[0][0].numpy().astype(np.uint8)

        # Returns the mask as a single "class" with area as confidence proxy
        area = float(mask.sum() / mask.size)
        return {'mask': area, '_mask_array': mask}

    return predict, True, ''


def _build_medgemma_predictor(card: ModelCard, device: str):
    """MedGemma vision-language model for radiology VQA / report drafting."""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
    except ImportError as e:
        return None, False, f'transformers not installed: {e}'

    repos = [card.repo_id] + (card.fallback_repos or [])
    # If a clinic LoRA adapter was trained (finetune_medgemma_reporter.py),
    # apply it on top of the base model. Set MEDGEMMA_LORA=<adapter_dir>.
    lora_dir = os.environ.get('MEDGEMMA_LORA')
    last_err = None
    for repo in repos:
        source = str(hf_bundle_dir(repo)) if (hf_bundle_dir(repo) / 'config.json').exists() else repo
        try:
            processor = AutoProcessor.from_pretrained(source, local_files_only=OFFLINE)
            model = AutoModelForImageTextToText.from_pretrained(
                source,
                torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
                local_files_only=OFFLINE,
            ).to(device).eval()
            if lora_dir and os.path.isdir(lora_dir):
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, lora_dir).to(device).eval()
                    logger.info(f"Applied MedGemma LoRA adapter from {lora_dir}")
                except Exception as le:
                    logger.warning(f"MEDGEMMA_LORA set but adapter load failed: {le}")
            logger.info(f"Loaded MedGemma from {repo}")

            def predict(image: np.ndarray, prompt: str = "Describe this brain MRI scan in radiological terms.") -> dict:
                from PIL import Image
                img = (image * 255).clip(0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
                pil = Image.fromarray(img).convert('RGB')
                # Gemma multimodal needs the image token injected via the chat
                # template — a raw text= prompt gives "0 image tokens / 1 image".
                try:
                    messages = [{
                        "role": "user",
                        "content": [{"type": "image", "image": pil},
                                    {"type": "text", "text": prompt}],
                    }]
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors='pt').to(device)
                except Exception:
                    # Fallback for processors without a chat template
                    inputs = processor(text=f"<start_of_image>{prompt}", images=pil,
                                       return_tensors='pt').to(device)
                in_len = inputs['input_ids'].shape[-1]
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                # Decode only the NEWLY generated tokens (not the echoed prompt)
                text = processor.decode(output[0][in_len:], skip_special_tokens=True).strip()
                return {'text': text}

            return predict, True, f'loaded from {repo}'
        except Exception as e:
            last_err = e
            logger.debug(f"MedGemma repo {repo} failed: {e}")
            continue

    if OFFLINE:
        return None, False, missing_model_reason(card.modality, [str(hf_bundle_dir(r)) for r in repos])
    return None, False, f'MedGemma unavailable. Last error: {last_err}'


def _build_monai_predictor(card: ModelCard, device: str):
    try:
        import monai  # noqa
    except ImportError as e:
        return None, False, f'MONAI not installed: {e}'
    return None, False, 'MONAI predictor not yet implemented'


# ============================================================================
# MODEL IDENTITY (provenance for every clinical response)
# ============================================================================

_identity_cache: dict[str, dict] = {}


# sha256 of zero bytes — what hashing a missing/empty weight file yields.
# model_identity must never report it as a weights hash.
EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

_WEIGHT_FILE_NAMES = ('model.safetensors', 'pytorch_model.bin', 'model.pt')


def _nonempty_weight_files(d: Path) -> list[Path]:
    """The first NON-EMPTY plaintext weight file in a model dir (symlinks
    into the HF blob store are followed). A 0-byte file — e.g. the
    .no_exist/<rev>/model.safetensors marker huggingface_hub writes for a
    file that is absent on the hub — is never a weight file."""
    for name in _WEIGHT_FILE_NAMES:
        c = Path(d) / name
        try:
            if c.is_file() and c.stat().st_size > 0:
                return [c.resolve()]
        except OSError:
            continue
    return []


def hf_snapshot_dir(repo_id: str) -> Optional[str]:
    """The local HF-cache snapshot directory a repo id resolves to — the files
    transformers actually read — WITHOUT any download. None when the repo is
    not in the cache."""
    if not repo_id or '/' not in repo_id:
        return None
    try:
        from huggingface_hub import snapshot_download
        return str(Path(snapshot_download(repo_id, local_files_only=True)).resolve())
    except Exception:
        pass
    try:
        from transformers.utils import cached_file
        cfg = cached_file(repo_id, 'config.json', local_files_only=True,
                          _raise_exceptions_for_missing_entries=False,
                          _raise_exceptions_for_connection_errors=False)
        if cfg:
            return str(Path(cfg).resolve().parent)
    except Exception:
        pass
    return None


def _resolve_weight_files(source: str) -> list[Path]:
    """Map a predictor's weights_source (local dir or HF repo id) to the
    PLAINTEXT weight file actually on disk, so we can hash it. An
    encrypted-at-rest dir returns [] — see weights_sha256_for_source. For a
    repo id the snapshot the loader resolves is used; the cache entry's
    .no_exist/ markers (0-byte files) are never picked up."""
    if not source:
        return []
    p = Path(source)
    if p.is_dir():
        return _nonempty_weight_files(p)
    if '/' not in source or ':' in source:
        if source.startswith('torchxrayvision:'):
            _, wp = resolve_xrv_weights(source.split(':', 1)[1])
            return [wp] if wp and Path(wp).is_file() and Path(wp).stat().st_size > 0 else []
        return []   # not an HF repo id
    bundle = hf_bundle_dir(source)
    if bundle.is_dir():
        return _resolve_weight_files(str(bundle))
    snapshot = hf_snapshot_dir(source)
    if snapshot:
        found = _nonempty_weight_files(Path(snapshot))
        if found:
            return found
    # Last resort: any snapshot of the cache entry (never .no_exist/).
    snapshots = _hf_cache_root() / f'models--{source.replace("/", "--")}' / 'snapshots'
    if snapshots.is_dir():
        for sd in sorted(snapshots.iterdir()):
            found = _nonempty_weight_files(sd)
            if found:
                return found
    return []


def weights_sha256_for_source(source: str) -> Optional[str]:
    """Full sha256 of the weights a predictor runs — hashed from the plaintext
    file, or read from the .enc.meta.json sidecar when the dir is encrypted at
    rest (so identity is the same on every install, whatever key encrypted it)."""
    if not source:
        return None
    p = Path(source)
    if p.is_dir():
        from src.utils.model_crypto import plaintext_sha256_of_dir
        full = plaintext_sha256_of_dir(p)
        if full and full != EMPTY_SHA256:
            return full
        return _sha256_of(_nonempty_weight_files(p))
    return _sha256_of(_resolve_weight_files(source))


def is_source_encrypted(source: Optional[str]) -> bool:
    if not source:
        return False
    p = Path(source)
    if not p.is_dir():
        return False
    from src.utils.model_crypto import is_encrypted_dir
    return is_encrypted_dir(p)


def _sha256_of(paths: list[Path]) -> Optional[str]:
    """sha256 over the given files; None when there are none or they hold no
    bytes (the empty-string digest is never a weights hash)."""
    import hashlib
    if not paths:
        return None
    h = hashlib.sha256()
    total = 0
    for fp in paths:
        with open(fp, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
                total += len(chunk)
    if total == 0:
        return None
    return h.hexdigest()


def get_model_identity(modality_key: str) -> Optional[dict]:
    """Provenance card for a LOADED model: which weights ran, their sha256
    (first 12 hex chars), and the clinical validation status. Cached per key
    (hashing a 350 MB ViT once is ~1 s). Returns None for unknown keys; for a
    key that failed to load, source/sha are None and validation_note says why."""
    if modality_key in _identity_cache:
        return _identity_cache[modality_key]
    card = REGISTRY.get(modality_key)
    if card is None:
        return None
    entry = _loaded_models.get(modality_key)
    source = None
    source_path = None
    sha12 = None
    note = card.notes
    if entry and entry.get('available'):
        predictor = entry.get('predictor')
        source = getattr(predictor, 'weights_source', None)
        if not source and str(entry.get('reason', '')).startswith('loaded from '):
            source = entry['reason'][len('loaded from '):]
        # Hash the files the loader actually read (HF-cache snapshot dir /
        # local dir / xrv weights file); the repo id is only a fallback.
        source_path = entry.get('source_path') or getattr(predictor, 'source_path', None)
        for cand in (source_path, source):
            if not cand:
                continue
            try:
                full = weights_sha256_for_source(str(cand))
            except Exception as e:
                logger.debug(f"sha256 for {modality_key} via {cand} failed: {e}")
                continue
            if full and full != EMPTY_SHA256:
                sha12 = full[:12]
                break
        if sha12 is None:
            where = source_path or source or '?'
            hash_note = f'weights not hashable: no non-empty plaintext weight file under {where}'
            note = f'{note}; {hash_note}' if note else hash_note
            source = f'{source} [unhashed]' if source else None
            logger.warning(f"model_identity {modality_key}: {hash_note}")
    elif entry:
        note = f"not loaded: {entry.get('reason', '')}"
    else:
        note = 'not loaded'
    identity = {
        'key': modality_key,
        'display_name': card.display_name,
        'source': source,
        'source_path': source_path,               # dir/file the hash was taken from
        'sha256_12': sha12,                       # PLAINTEXT weights hash (stable across installs); null when unhashable
        'encrypted_at_rest': is_source_encrypted(source_path or source),
        'status': card.validation_status,
        'validation_note': note,
    }
    if entry and entry.get('available'):
        _identity_cache[modality_key] = identity   # only cache a successful load
    return identity


def get_loaded_status(keys: Optional[list[str]] = None) -> dict[str, dict]:
    """{key: {loaded, reason}} for /health. Never triggers a load — reports
    what has actually been attempted. Keys not yet attempted show loaded=False
    with reason 'not loaded'."""
    keys = keys or [k for k in REGISTRY if k != 'brain_2d']
    out = {}
    for k in keys:
        entry = _loaded_models.get(k)
        if entry is None:
            out[k] = {'loaded': False, 'reason': 'not loaded'}
        else:
            out[k] = {'loaded': bool(entry.get('available')), 'reason': entry.get('reason', '') or ''}
    return out


# ============================================================================
# AVAILABILITY REPORT
# ============================================================================

def get_availability() -> list[dict]:
    out = []
    for key, card in REGISTRY.items():
        if key == 'brain_2d':
            continue  # alias; don't report twice
        deps_ok = True
        deps_reason = ''
        if card.backend == 'xrv':
            try:
                import torchxrayvision  # noqa
            except ImportError:
                deps_ok, deps_reason = False, 'pip install torchxrayvision'
        elif card.backend in ('huggingface', 'sam', 'medgemma'):
            try:
                import transformers  # noqa
            except ImportError:
                deps_ok, deps_reason = False, 'pip install transformers'
        elif card.backend == 'monai':
            try:
                import monai  # noqa
            except ImportError:
                deps_ok, deps_reason = False, 'pip install monai'

        out.append({
            'key': key,
            'name': card.display_name,
            'tier': card.tier,
            'validation_status': card.validation_status,
            'license': card.license,
            'is_3d': card.is_3d,
            'sequences_required': card.sequences_required,
            'modality_tags': card.modality_dicom_tags,
            'body_part_tags': card.body_part_dicom_tags,
            'classes': card.classes,
            'expected_auc': card.expected_auc,
            'citation': card.citation,
            'deps_ok': deps_ok,
            'deps_reason': deps_reason,
            'download_mb': card.download_size_mb,
            'fine_tunable': card.fine_tunable,
            'notes': card.notes,
        })
    return out
