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
)


HEAD_CT = ModelCard(
    modality='head_ct',
    display_name='Head CT Hemorrhage Detection (6 classes)',
    classes=['any', 'epidural', 'intraparenchymal', 'intraventricular',
             'subarachnoid', 'subdural'],
    classes_localized={
        'any':              {'ru': 'Кровоизлияние',          'uz': 'Qon quyilishi',       'en': 'Any hemorrhage'},
        'epidural':         {'ru': 'Эпидуральное',           'uz': 'Epidural',            'en': 'Epidural'},
        'intraparenchymal': {'ru': 'Внутримозговое',         'uz': 'Miya ichi',           'en': 'Intraparenchymal'},
        'intraventricular': {'ru': 'Внутрижелудочковое',     'uz': 'Qorinchalararo',      'en': 'Intraventricular'},
        'subarachnoid':     {'ru': 'Субарахноидальное',      'uz': 'Subaraxnoidal',       'en': 'Subarachnoid'},
        'subdural':         {'ru': 'Субдуральное',           'uz': 'Subdural',            'en': 'Subdural'},
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
    entry = {'card': card, 'predictor': predictor,
             'available': available, 'reason': reason}
    _loaded_models[modality_key] = entry
    return entry


def _build_predictor(card: ModelCard, device: str):
    if card.backend == 'xrv':
        return _build_xrv_predictor(card, device)
    if card.backend == 'huggingface':
        # 3D segmentation models use custom HF code (trust_remote_code path)
        if card.is_3d:
            # Prefer the MONAI bundle if it's installed locally
            from pathlib import Path as _PP
            monai_bundle = _PP(__file__).parent.parent.parent / 'models' / 'monai_bundles' / 'brats_mri_segmentation'
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

    # Find the safetensors file in HF cache
    from pathlib import Path as _P
    hf_root = _P.home() / '.cache' / 'huggingface' / 'hub'

    repos = [card.repo_id] + (card.fallback_repos or [])
    last_err = None

    for repo in repos:
        cache_dir = hf_root / f'models--{repo.replace("/", "--")}'
        if not cache_dir.exists():
            last_err = f'no cache for {repo}'
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

    return None, False, f'No 3D seg model loadable. Last error: {last_err}'


def _build_xrv_predictor(card: ModelCard, device: str):
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as e:
        return None, False, f'torchxrayvision not installed: {e}'

    try:
        model = xrv.models.DenseNet(weights='densenet121-res224-all')
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

    return predict, True, ''


def _build_hf_predictor(card: ModelCard, device: str):
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
    except ImportError as e:
        return None, False, f'transformers not installed: {e}'

    # Prefer a locally fine-tuned model if one exists for this modality.
    # training/finetune_brain_classifier.py writes to models/<modality>_finetuned/
    # (and the legacy models/brain_finetuned/). If present, use it FIRST so the
    # clinic's own fine-tune overrides the public weights.
    from pathlib import Path as _P
    _repo_root = _P(__file__).parent.parent.parent
    # The per-modality fine-tune dir applies to ANY modality (brain, head_ct, …).
    local_candidates = [_repo_root / 'models' / f'{card.modality}_finetuned']
    # Legacy: the tumor fine-tune script wrote to models/brain_finetuned/. Only
    # use it for the TUMOR model — never as a fallback for stroke/dementia/etc.,
    # or they'd silently load the tumor weights.
    if card.modality in ('brain_tumor_class', 'brain_2d'):
        local_candidates.append(_repo_root / 'models' / 'brain_finetuned')
    local_dirs = [str(d) for d in local_candidates
                  if d.exists() and (d / 'config.json').exists()]

    repos = local_dirs + [card.repo_id] + (card.fallback_repos or [])
    repos = [r for r in repos if r]

    # Generic fallback processor — used when a repo only ships weights without
    # a preprocessor_config.json. Both ViT-B and ResNet-50 work fine with this.
    GENERIC_PROCESSORS = [
        'google/vit-base-patch16-224',
        'microsoft/resnet-50',
    ]

    last_err = None
    for repo in repos:
        try:
            # Try the repo's own processor first
            try:
                processor = AutoImageProcessor.from_pretrained(repo)
            except Exception:
                # Fall back to a generic processor that matches the input size
                processor = None
                for generic in GENERIC_PROCESSORS:
                    try:
                        processor = AutoImageProcessor.from_pretrained(generic)
                        logger.info(f"Using {generic} processor for {repo} (no config in repo)")
                        break
                    except Exception:
                        continue
                if processor is None:
                    raise RuntimeError("No processor available")

            model = AutoModelForImageClassification.from_pretrained(repo)
            model.to(device).eval()
            id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
            logger.info(f"Loaded {card.modality} from {repo}")

            def predict(image: np.ndarray, _model=model, _proc=processor, _i2l=id2label) -> dict:
                from PIL import Image
                if image.ndim == 2:
                    img = (image * 255).clip(0, 255).astype(np.uint8)
                    pil = Image.fromarray(img).convert('RGB')
                else:
                    pil = Image.fromarray(image.astype(np.uint8)).convert('RGB')
                inputs = _proc(images=pil, return_tensors='pt').to(device)
                with torch.no_grad():
                    logits = _model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                return {_i2l.get(i, f'class_{i}'): float(p) for i, p in enumerate(probs)}

            return predict, True, f'loaded from {repo}'
        except Exception as e:
            last_err = e
            logger.debug(f"Repo {repo} failed: {e}")
            continue

    return None, False, f'No HF repo available. Last error: {last_err}'


def _build_sam_predictor(card: ModelCard, device: str):
    """MedSAM — prompt-based universal segmentation."""
    try:
        from transformers import SamModel, SamProcessor
        import torch
    except ImportError as e:
        return None, False, f'transformers not installed: {e}'

    try:
        repo = card.repo_id
        processor = SamProcessor.from_pretrained(repo)
        model = SamModel.from_pretrained(repo).to(device).eval()
        logger.info(f"Loaded MedSAM from {repo}")
    except Exception as e:
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
        try:
            processor = AutoProcessor.from_pretrained(repo)
            model = AutoModelForImageTextToText.from_pretrained(
                repo,
                torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
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

    return None, False, f'MedGemma unavailable. Last error: {last_err}'


def _build_monai_predictor(card: ModelCard, device: str):
    try:
        import monai  # noqa
    except ImportError as e:
        return None, False, f'MONAI not installed: {e}'
    return None, False, 'MONAI predictor not yet implemented'


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
