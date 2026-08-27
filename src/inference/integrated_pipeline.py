"""
================================================================================
  SENTINEL MEDICAL AI — INTEGRATED INFERENCE PIPELINE
================================================================================
  Full end-to-end medical AI pipeline combining:

  1. DenseNet121    → Pathology classification (Pneumonia 87%, Effusion 62%...)
  2. Grad-CAM       → Visual heatmap showing WHERE the pathology is
  3. nnU-Net        → Pixel-level segmentation masks (Phase 2)
  4. Gemma 3 VL     → Natural language report generation (Russian/Uzbek/English)

  FLOW:
    DICOM image
        ↓
    ┌─ DenseNet121 ─────────┐  "Pneumonia 87%, Effusion 62%"
    │                       │
    │   + Grad-CAM         │  [Heatmap image]
    │                       │
    │   + nnU-Net (opt)     │  [Segmentation mask]
    └───────────────────────┘
        ↓ findings + heatmap
    ┌─ Gemma 3 VL ──────────┐
    │                       │
    │   System: "You're a  │
    │   radiologist..."    │
    │                       │
    │   Input: image +      │
    │          findings    │
    │                       │
    │   Output: Full RU/UZ │
    │   structured report  │
    └───────────────────────┘
        ↓
    Radiologist reviews & signs → PDF
================================================================================
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class IntegratedResult:
    """Complete output from integrated AI pipeline."""
    # Classification
    findings: list  # List of (class, confidence) tuples
    is_normal: bool

    # Visualization
    heatmap_image: Optional[np.ndarray] = None  # (H, W, 3) RGB
    segmentation_masks: Optional[dict] = None  # {class_name: mask}

    # Report
    structured_report: Optional[dict] = None  # {indication, technique, description, ...}
    natural_report: Optional[str] = None      # Full natural language report

    # Metadata
    inference_time_ms: int = 0
    densenet_time_ms: int = 0
    gemma_time_ms: int = 0
    model_versions: dict = None


class IntegratedPipeline:
    """Complete medical imaging AI pipeline.

    Usage:
        pipeline = IntegratedPipeline(
            densenet_path='models/densenet/best_model.pt',
            gemma_path='models/gemma_medical',  # optional
        )
        result = pipeline.analyze('chest.dcm', language='ru')
        print(result.natural_report)
    """

    def __init__(
        self,
        densenet_path: str,
        gemma_path: Optional[str] = None,
        nnunet_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or self._auto_device()

        # Load DenseNet121 (required)
        logger.info(f"Loading DenseNet121 from {densenet_path}...")
        self.densenet, self.class_names, self.thresholds = self._load_densenet(densenet_path)
        self.gradcam = self._setup_gradcam()

        # Load Gemma (optional - for Phase 3 report generation)
        self.gemma = None
        self.gemma_processor = None
        if gemma_path and Path(gemma_path).exists():
            logger.info(f"Loading Gemma 3 VL from {gemma_path}...")
            self.gemma, self.gemma_processor = self._load_gemma(gemma_path)
        else:
            logger.info("Gemma not loaded — using template reports only")

        # Load nnU-Net (optional - for segmentation)
        self.nnunet = None
        if nnunet_path and Path(nnunet_path).exists():
            logger.info(f"Loading nnU-Net from {nnunet_path}...")
            # nnU-Net loading... (Phase 2)

    def _auto_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _load_densenet(self, path: str):
        """Load trained DenseNet121 classifier."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.training.densenet_trainer import DenseNet121Classifier

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        class_names = ckpt.get('class_names', [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia',
        ])
        thresholds = ckpt.get('optimal_thresholds', [0.5] * len(class_names))

        model = DenseNet121Classifier(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model, class_names, thresholds

    def _setup_gradcam(self):
        """Initialize Grad-CAM for heatmap generation."""
        from src.training.gradcam import GradCAM
        return GradCAM(self.densenet, target_layer_name='densenet.features.denseblock4')

    def _load_gemma(self, path: str):
        """Load fine-tuned Gemma 3 VL model."""
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            from peft import PeftModel

            base_model_name = 'google/gemma-3-4b-it'  # default base

            # Check if this is a LoRA adapter or full model
            adapter_config = Path(path) / 'adapter_config.json'
            if adapter_config.exists():
                # It's a LoRA adapter
                base = Gemma3ForConditionalGeneration.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )
                model = PeftModel.from_pretrained(base, path)
            else:
                # Full model
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )

            processor = AutoProcessor.from_pretrained(path)
            model.eval()
            return model, processor

        except Exception as e:
            logger.warning(f"Failed to load Gemma: {e}. Will use templates only.")
            return None, None

    # ===========================================================
    # MAIN INFERENCE
    # ===========================================================

    def analyze(
        self,
        dicom_path: str,
        language: str = 'ru',
        generate_natural_report: bool = True,
    ) -> IntegratedResult:
        """Run full integrated analysis on a DICOM file.

        Args:
            dicom_path: Path to DICOM file
            language: 'ru', 'uz', or 'en' for report language
            generate_natural_report: Use Gemma for natural language (slower)

        Returns:
            IntegratedResult with findings, heatmap, and report
        """
        import time
        from src.pipeline.dicom_loader import load_dicom
        from src.pipeline.preprocessor import preprocess_study

        start_time = time.time()

        # 1. Load + preprocess DICOM
        study = load_dicom(dicom_path)
        if study is None:
            raise ValueError(f"Failed to load DICOM: {dicom_path}")

        processed_image = preprocess_study(study, target_size=512)
        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. DenseNet121 classification
        densenet_start = time.time()
        with torch.no_grad():
            logits = self.densenet(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        densenet_time = int((time.time() - densenet_start) * 1000)

        # Apply optimized thresholds
        findings = []
        for i, (name, prob) in enumerate(zip(self.class_names, probs)):
            threshold = self.thresholds[i] if isinstance(self.thresholds, (list, np.ndarray)) else 0.5
            if prob >= threshold:
                findings.append({
                    'class_name': name,
                    'confidence': float(prob),
                    'threshold': float(threshold),
                })
        findings.sort(key=lambda f: f['confidence'], reverse=True)
        is_normal = len(findings) == 0

        # 3. Grad-CAM heatmap for top finding
        heatmap_image = None
        if not is_normal:
            top_class_idx = self.class_names.index(findings[0]['class_name'])
            cam = self.gradcam.generate(input_tensor, target_class=top_class_idx)
            heatmap_image = self.gradcam.generate_overlay(processed_image, cam)

        # 4. Build findings text for Gemma
        if is_normal:
            findings_text = "AI analysis: No significant pathology detected."
        else:
            finding_strs = [
                f"{f['class_name']} ({f['confidence']*100:.0f}%)"
                for f in findings[:5]
            ]
            findings_text = f"DenseNet121 findings: {', '.join(finding_strs)}"

        # 5. Generate structured template report (always)
        structured_report = self._generate_template_report(findings, language, study)

        # 6. Generate natural language report with Gemma (optional)
        natural_report = None
        gemma_time = 0
        if generate_natural_report and self.gemma is not None:
            gemma_start = time.time()
            natural_report = self._gemma_generate(
                processed_image, findings_text, language
            )
            gemma_time = int((time.time() - gemma_start) * 1000)

        total_time = int((time.time() - start_time) * 1000)

        return IntegratedResult(
            findings=findings,
            is_normal=is_normal,
            heatmap_image=heatmap_image,
            segmentation_masks=None,
            structured_report=structured_report,
            natural_report=natural_report,
            inference_time_ms=total_time,
            densenet_time_ms=densenet_time,
            gemma_time_ms=gemma_time,
            model_versions={
                'densenet': 'DenseNet121-v1.0',
                'gemma': 'Gemma3-4B-medical' if self.gemma else None,
            },
        )

    # ===========================================================
    # GEMMA REPORT GENERATION
    # ===========================================================

    def _gemma_generate(self, image, findings_text, language='ru'):
        """Generate natural report using Gemma 3 VL."""
        from PIL import Image

        # Convert numpy to PIL
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8).convert('RGB')

        # Prompt
        prompts = {
            'ru': (
                "Ты — опытный врач-рентгенолог. На основе рентгеновского снимка и "
                "результатов ИИ-анализа составь структурированный протокол.",
                f"Результаты ИИ-анализа:\n{findings_text}\n\nСоставь полный протокол.",
            ),
            'uz': (
                "Siz tajribali rentgenolog shifokorsiz. Rentgen suratga va AI natijalariga "
                "asoslanib, strukturaviy bayonnoma tuzing.",
                f"AI-tahlil natijalari:\n{findings_text}\n\nTo'liq bayonnoma tuzing.",
            ),
            'en': (
                "You are an experienced radiologist. Based on the X-ray and AI findings, "
                "write a structured radiology report.",
                f"AI findings:\n{findings_text}\n\nWrite a complete report.",
            ),
        }
        system, user = prompts.get(language, prompts['ru'])

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{system}\n\n{user}"},
            ],
        }]

        text = self.gemma_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.gemma_processor(
            text=text, images=pil_image, return_tensors="pt"
        ).to(self.gemma.device)

        with torch.no_grad():
            outputs = self.gemma.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        generated = outputs[0][inputs['input_ids'].shape[1]:]
        return self.gemma_processor.decode(generated, skip_special_tokens=True).strip()

    # ===========================================================
    # TEMPLATE FALLBACK (when Gemma not available)
    # ===========================================================

    def _generate_template_report(self, findings, language, study):
        """Generate structured template report (doesn't need Gemma)."""
        if not findings:
            # Normal report
            templates = {
                'ru': {
                    'description': 'Легочные поля прозрачны, без очаговых и инфильтративных изменений. Корни легких структурны. Сердечная тень обычной формы. Диафрагма расположена обычно.',
                    'conclusion': 'Патологических изменений не выявлено.',
                    'recommendation': 'Контроль по клиническим показаниям.',
                },
                'uz': {
                    'description': "O'pka maydonlari tiniq, o'choqli va infiltrativ o'zgarishlarsiz. O'pka ildizlari strukturali. Yurak soyasi odatiy shaklda.",
                    'conclusion': "Patologik o'zgarishlar aniqlanmadi.",
                    'recommendation': "Klinik ko'rsatmalarga muvofiq nazorat.",
                },
                'en': {
                    'description': 'Lung fields are clear, without focal or infiltrative changes. Hilar structures are preserved. Cardiac silhouette is normal in size and shape.',
                    'conclusion': 'No pathological changes detected.',
                    'recommendation': 'Follow-up as clinically indicated.',
                },
            }
            tmpl = templates.get(language, templates['ru'])
            return {
                'indication': 'Plановое обследование' if language == 'ru' else 'Routine examination',
                'technique': 'Рентгенография в прямой проекции' if language == 'ru' else 'PA chest radiograph',
                **tmpl,
                'findings_list': [],
            }

        # Build description from findings
        descriptions_by_class = {
            'Pneumonia': {
                'ru': 'определяется участок инфильтрации легочной ткани',
                'uz': "o'pka to'qimasining infiltratsiya sohasi aniqlanadi",
                'en': 'an area of lung infiltration is identified',
            },
            'Effusion': {
                'ru': 'наличие жидкости в плевральной полости',
                'uz': "plevral bo'shliqda suyuqlik mavjudligi",
                'en': 'pleural effusion present',
            },
            'Cardiomegaly': {
                'ru': 'увеличение размеров сердечной тени',
                'uz': "yurak soyasi o'lchamlari kattalashgan",
                'en': 'enlarged cardiac silhouette',
            },
            # Add more...
        }

        descriptions = []
        finding_names = []
        for f in findings:
            desc = descriptions_by_class.get(f['class_name'], {}).get(language, f['class_name'])
            descriptions.append(f"{desc} (достоверность: {f['confidence']*100:.0f}%)")
            finding_names.append(f['class_name'])

        if language == 'ru':
            description = '. '.join(descriptions) + '.'
            conclusion = f"Рентгенологическая картина соответствует: {', '.join(finding_names)}."
            recommendation = 'Динамическое наблюдение. Консультация специалиста.'
        elif language == 'uz':
            description = '. '.join(descriptions) + '.'
            conclusion = f"Rentgenologik ko'rinish: {', '.join(finding_names)}."
            recommendation = "Dinamik kuzatuv. Mutaxassis maslahati."
        else:
            description = '. '.join(descriptions) + '.'
            conclusion = f"Radiographic findings consistent with: {', '.join(finding_names)}."
            recommendation = 'Clinical correlation and follow-up recommended.'

        return {
            'indication': 'Обследование органов грудной клетки' if language == 'ru' else 'Chest examination',
            'technique': 'Рентгенография в прямой проекции' if language == 'ru' else 'PA chest radiograph',
            'description': description,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'findings_list': [f['class_name'] for f in findings],
        }
