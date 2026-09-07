"""
================================================================================
  SENTINEL MEDICAL AI — GEMMA 3 REPORT ENGINE
================================================================================
  Generates natural Russian / Uzbek / English radiology reports using Gemma 3.

  RUNS LOCALLY — NO INTERNET NEEDED (privacy-safe for clinic data).

  ARCHITECTURE:
    Image findings (from DenseNet/Brain model)
            ↓
    Structured prompt with medical context
            ↓
    Gemma 3 (via Ollama) running locally
            ↓
    Natural language report in RU/UZ/EN
            ↓
    Doctor reviews + edits + signs

  BACKENDS SUPPORTED:
    1. Ollama (RECOMMENDED — 100% local, free, works offline)
       Install: https://ollama.com
       Pull: ollama pull gemma3:4b
       Runs on GTX 1650 with 4GB VRAM

    2. HuggingFace Transformers (fallback)
       pip install transformers accelerate
       Downloads model to ~/.cache/huggingface

    3. Google AI Studio API (online, free tier, for testing only)
       pip install google-generativeai
       Not recommended for production (patient data privacy)

  USAGE:
    from src.inference.gemma_report_engine import GemmaReportEngine
    engine = GemmaReportEngine(backend='ollama')

    report = engine.generate(
        findings=[
            {"class_name": "Pneumonia", "confidence": 0.87, "location": "Right lower"},
            {"class_name": "Effusion", "confidence": 0.62, "location": "Right lower"},
        ],
        language='ru',
        patient_info={"id": "P-001", "age": 45, "sex": "M"},
    )
================================================================================
"""

import json
import requests
from typing import Optional
from dataclasses import dataclass, field
# field is used for default_factory below
from loguru import logger


# ============================================================================
# MEDICAL SYSTEM PROMPTS — Multilingual
# ============================================================================

SYSTEM_PROMPTS = {
    'ru': """Ты — опытный врач-рентгенолог с 20-летним стажем. Ты работаешь в клинике в Ташкенте.

На основе результатов ИИ-анализа медицинского изображения составь структурированный протокол рентгенологического исследования на русском языке.

Структура протокола (ОБЯЗАТЕЛЬНО используй именно эти разделы):
1. КЛИНИЧЕСКОЕ ПОКАЗАНИЕ
2. МЕТОДИКА
3. ОПИСАНИЕ (подробное описание всех патологических находок)
4. ЗАКЛЮЧЕНИЕ (диагностический вывод, краткий но точный)
5. РЕКОМЕНДАЦИИ (конкретные действия для лечащего врача)

Требования:
- Используй стандартную медицинскую терминологию
- Будь точным и объективным
- Указывай локализацию патологий (правая/левая сторона, доля)
- Включай процент достоверности для каждой находки
- Не добавляй ничего, чего нет в данных ИИ-анализа
- Пиши кратко и по существу""",

    'uz': """Siz 20 yillik tajribaga ega tajribali rentgenolog shifokorsiz. Siz Toshkentdagi klinikada ishlaysiz.

Tibbiy rasm AI-tahlili natijalari asosida strukturaviy rentgenologik tekshiruv bayonnomasini o'zbek tilida tuzing.

Bayonnoma tuzilishi (ALBATTA quyidagi bo'limlardan foydalaning):
1. KLINIK KO'RSATMA
2. METODIKA
3. TAVSIF (barcha patologik topilmalarning batafsil tavsifi)
4. XULOSA (tashxis — qisqa, lekin aniq)
5. TAVSIYALAR (davolovchi shifokor uchun aniq harakatlar)

Talablar:
- Standart tibbiy terminologiyadan foydalaning
- Aniq va ob'ektiv bo'ling
- Patologiyalarning joylashuvini ko'rsating (o'ng/chap tomon, ulush)
- Har bir topilma uchun ishonchlilik foizini kiriting
- AI-tahlil ma'lumotlarida yo'q narsani qo'shmang
- Qisqa va aniq yozing""",

    'en': """You are an experienced radiologist with 20 years of experience, working at a clinic in Tashkent.

Based on the AI analysis results of a medical image, generate a structured radiology report in English.

Report structure (use EXACTLY these sections):
1. CLINICAL INDICATION
2. TECHNIQUE
3. FINDINGS (detailed description of all pathological findings)
4. IMPRESSION (diagnostic conclusion — concise but precise)
5. RECOMMENDATIONS (specific actions for the referring physician)

Requirements:
- Use standard medical terminology
- Be precise and objective
- Indicate pathology location (right/left side, lobe)
- Include confidence percentage for each finding
- Do not add anything not present in the AI analysis data
- Write concisely and to the point""",
}


USER_PROMPT_TEMPLATES = {
    'ru': """ДАННЫЕ ИИ-АНАЛИЗА:
Модальность: {modality}
Область исследования: {body_part}
Пациент: ID {patient_id}, возраст {age}, пол {sex}
Дата исследования: {study_date}

НАЙДЕННЫЕ ПАТОЛОГИИ:
{findings_list}

Общий статус: {overall_status}

Составь полный рентгенологический протокол используя структуру выше.""",

    'uz': """AI-TAHLIL NATIJALARI:
Modallik: {modality}
Tekshiruv sohasi: {body_part}
Bemor: ID {patient_id}, yosh {age}, jinsi {sex}
Tekshiruv sanasi: {study_date}

ANIQLANGAN PATOLOGIYALAR:
{findings_list}

Umumiy holat: {overall_status}

Yuqoridagi tuzilishdan foydalanib to'liq rentgenologik bayonnoma tuzing.""",

    'en': """AI ANALYSIS RESULTS:
Modality: {modality}
Body part: {body_part}
Patient: ID {patient_id}, age {age}, sex {sex}
Study date: {study_date}

DETECTED PATHOLOGIES:
{findings_list}

Overall status: {overall_status}

Generate a full radiology report using the structure above.""",
}


def _cloud_llm_allowed() -> bool:
    """Cloud (Google AI) backend is opt-in only — never on by default."""
    import os
    return os.environ.get('SENTINEL_ALLOW_CLOUD_LLM') == '1'


# ============================================================================
# REPORT ENGINE
# ============================================================================

@dataclass
class GemmaReportEngine:
    """Medical report generator using Gemma 3.

    Backends:
    - 'ollama'        — Local (Ollama) — best for clinic deployment, free
    - 'google_ai'     — Cloud (Google AI Studio) — free tier, no laptop GPU needed
    - 'transformers'  — Local (HuggingFace) — uses laptop GPU
    - 'auto'          — Tries ollama → google_ai → templates
    """

    backend: str = 'auto'             # 'ollama' | 'google_ai' | 'transformers' | 'auto'
    # Model tags to try in order (newest first):
    #   gemma4:e4b   — Released April 2026 (4B, optimized for laptops)
    #   gemma4:e2b   — Smaller (1.5GB RAM)
    #   gemma3:4b    — Stable fallback
    #   gemma3:12b   — Higher quality
    model_name: str = 'gemma4:e4b'    # Try Gemma 4 first, falls back to gemma3:4b
    temperature: float = 0.3          # Low for medical accuracy
    max_tokens: int = 1024

    # Ollama settings
    ollama_url: str = 'http://localhost:11434'

    # Google AI Studio settings (free tier — get key at aistudio.google.com)
    google_api_key: Optional[str] = None  # Set via env: GOOGLE_AI_KEY
    # Auto-falls back through these models (newest first):
    google_model: str = 'gemini-2.5-flash'  # Latest free-tier model
    google_model_fallbacks: list = field(default_factory=lambda: [
        'gemini-2.5-flash',       # Best free tier (April 2026)
        'gemini-2.0-flash',       # Stable
        'gemini-flash-latest',    # Always points to newest flash
        'gemini-2.0-flash-exp',   # Experimental
        'gemini-1.5-flash-002',   # Legacy backup
        'gemini-pro-latest',      # Pro tier (rate-limited)
    ])

    # Internal
    _model: Optional[object] = None
    _tokenizer: Optional[object] = None
    _google_client: Optional[object] = None

    def __post_init__(self):
        # Auto-detect best available backend. The cloud backend is opt-in ONLY
        # (SENTINEL_ALLOW_CLOUD_LLM=1): an on-prem clinic must never have
        # report prompts leave the building by accident.
        if self.backend == 'auto':
            if self._check_ollama():
                self.backend = 'ollama'
            elif _cloud_llm_allowed() and self._check_google_ai():
                self.backend = 'google_ai'
                logger.info("Using Google AI Studio (cloud, free tier) — SENTINEL_ALLOW_CLOUD_LLM=1")
            else:
                self.backend = 'template'
                logger.warning("No local AI backend available — using templates only")
            return

        if self.backend == 'ollama':
            self._check_ollama()
        elif self.backend == 'google_ai':
            if not _cloud_llm_allowed():
                logger.warning(
                    "backend='google_ai' requested but SENTINEL_ALLOW_CLOUD_LLM is not 1 — "
                    "cloud LLM disabled, using templates")
                self.backend = 'template'
            else:
                self._check_google_ai()
        elif self.backend == 'transformers':
            self._load_transformers()

    def _check_google_ai(self) -> bool:
        """Check if Google AI Studio is available (needs API key AND the
        SENTINEL_ALLOW_CLOUD_LLM=1 opt-in).

        Tries new google-genai SDK first, falls back to old google-generativeai.
        """
        import os
        if not _cloud_llm_allowed():
            return False
        api_key = self.google_api_key or os.environ.get('GOOGLE_AI_KEY')
        if not api_key:
            logger.debug("No GOOGLE_AI_KEY set — Google AI unavailable")
            return False

        # Try new SDK first (google-genai)
        try:
            from google import genai
            self._google_client = genai.Client(api_key=api_key)
            self._google_sdk = 'new'
            # Try to find a working model
            for model in self.google_model_fallbacks:
                try:
                    test = self._google_client.models.generate_content(
                        model=model, contents="test", config={'max_output_tokens': 5},
                    )
                    self.google_model = model
                    logger.info(f"Google AI ready ({model}, new SDK)")
                    return True
                except Exception:
                    continue
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"New Google SDK init failed: {e}")

        # Fall back to old SDK (google-generativeai) — try each model
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            for model_name in self.google_model_fallbacks:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    # Quick test
                    resp = test_model.generate_content(
                        "test",
                        generation_config={'max_output_tokens': 5},
                    )
                    if resp:
                        self._google_client = test_model
                        self._google_sdk = 'old'
                        self.google_model = model_name
                        logger.info(f"Google AI ready ({model_name}, legacy SDK)")
                        return True
                except Exception as e:
                    logger.debug(f"Model {model_name} failed: {str(e)[:80]}")
                    continue

            # If none worked, list what IS available
            logger.warning("None of the preferred models worked. Available models:")
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        logger.warning(f"  → {m.name}")
            except Exception:
                pass
            return False

        except ImportError:
            logger.debug("google-generativeai not installed")
            return False
        except Exception as e:
            logger.debug(f"Google AI init failed: {e}")
            return False

    def _generate_google_ai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate via Google AI Studio API (auto-handles new vs old SDK)."""
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

            # New SDK (google-genai)
            if getattr(self, '_google_sdk', 'old') == 'new':
                response = self._google_client.models.generate_content(
                    model=self.google_model,
                    contents=full_prompt,
                    config={
                        'temperature': self.temperature,
                        'max_output_tokens': self.max_tokens,
                        'top_p': 0.9,
                    },
                )
                return response.text.strip()

            # Old SDK (google-generativeai) - still works
            response = self._google_client.generate_content(
                full_prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                    'top_p': 0.9,
                },
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Google AI generation failed: {e}")
            return ""

    # ------------------------------------------------------------
    # OLLAMA BACKEND (RECOMMENDED)
    # ------------------------------------------------------------

    def _check_ollama(self):
        """Verify Ollama is running and model is available.

        Auto-falls back through preferred model list:
        gemma4:e4b → gemma3:12b → gemma3:4b → gemma3:1b
        """
        try:
            resp = requests.get(f'{self.ollama_url}/api/tags', timeout=3)
            if resp.status_code != 200:
                return False

            available_models = [m['name'] for m in resp.json().get('models', [])]
            if not available_models:
                logger.warning("Ollama running but no models pulled")
                return False

            # Try preferred models in order
            preferred = [
                self.model_name,        # User's first choice
                'gemma4:e4b',           # Gemma 4 (April 2026)
                'gemma4:e2b',           # Smaller Gemma 4
                'gemma3:12b',           # Larger Gemma 3
                'gemma3:4b',            # Standard
                'gemma3:1b',            # Smallest
                'gemma2:9b',            # Older fallback
                'llama3.1:8b',          # Llama fallback
            ]

            for candidate in preferred:
                base = candidate.split(':')[0]
                tag = candidate.split(':')[1] if ':' in candidate else 'latest'
                # Match exact tag OR base name
                for available in available_models:
                    if available == candidate or available.startswith(f"{base}:"):
                        self.model_name = available
                        logger.info(f"Ollama ready with {self.model_name}")
                        return True

            # Nothing matched — report what's available
            logger.warning(
                f"Ollama has these models: {available_models}. "
                f"Pull a Gemma model: ollama pull gemma4:e4b"
            )
            return False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Install from https://ollama.com")
            return False

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama API for generation."""
        try:
            resp = requests.post(
                f'{self.ollama_url}/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': user_prompt,
                    'system': system_prompt,
                    'stream': False,
                    'options': {
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens,
                        'top_p': 0.9,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()['response'].strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            # Return empty so generate()'s outer fallback runs with the REAL
            # findings + correct language (not a hardcoded Russian "Норма").
            return ""

    # ------------------------------------------------------------
    # HUGGINGFACE TRANSFORMERS BACKEND (fallback)
    # ------------------------------------------------------------

    def _load_transformers(self):
        """Load Gemma 3 via HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info("Loading Gemma 3 via HuggingFace (one-time download ~5GB)...")
            model_id = 'google/gemma-3-4b-it'
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map='auto',
            )
            logger.info("Gemma 3 loaded successfully")
        except Exception as e:
            logger.error(f"Transformers load failed: {e}")
            self._model = None

    def _generate_transformers(self, system_prompt: str, user_prompt: str) -> str:
        """Generate via HuggingFace model."""
        if self._model is None:
            return "ERROR: Model not loaded"
        import torch

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors='pt').to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
            )
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------
    # MAIN API
    # ------------------------------------------------------------

    def generate(
        self,
        findings: list[dict],
        language: str = 'ru',
        patient_info: Optional[dict] = None,
        modality: str = 'CR',
        body_part: str = 'CHEST',
        study_date: str = '',
    ) -> str:
        """Generate a full radiology report.

        Args:
            findings: List of dicts like [{"class_name": "Pneumonia", "confidence": 0.87, "location": "Right lower"}]
            language: 'ru' | 'uz' | 'en'
            patient_info: Optional dict {"id", "age", "sex"}
            modality: CR/DX/CT/MR/etc.
            body_part: CHEST/BRAIN/etc.
            study_date: YYYY-MM-DD

        Returns:
            Full structured report as string
        """
        language = language if language in SYSTEM_PROMPTS else 'ru'

        # Brain studies → use the deterministic, professionally-worded, fully
        # multilingual brain templates rather than the small LLM. They are
        # cleaner and never emit conversational preambles or placeholders,
        # which matters for clinical reports. (Gemma is still used for chat Q&A.)
        _bp = (body_part or '').upper()
        _mod = (modality or '').upper()
        if 'BRAIN' in _bp or 'HEAD' in _bp or ('MR' in _mod and not _bp):
            try:
                from src.inference.brain_report_templates import build_brain_report
                sec = build_brain_report(findings, language=language, modality=modality)
                labels = {
                    'ru': ['КЛИНИЧЕСКОЕ ПОКАЗАНИЕ', 'МЕТОДИКА', 'ОПИСАНИЕ', 'ЗАКЛЮЧЕНИЕ', 'РЕКОМЕНДАЦИИ'],
                    'uz': ['KLINIK KO‘RSATMA', 'METODIKA', 'TAVSIF', 'XULOSA', 'TAVSIYALAR'],
                    'en': ['CLINICAL INDICATION', 'TECHNIQUE', 'DESCRIPTION', 'IMPRESSION', 'RECOMMENDATIONS'],
                }.get(language, None)
                keys = ['clinical_indication', 'technique', 'description', 'impression', 'recommendation']
                if labels and isinstance(sec, dict):
                    report = "\n\n".join(
                        f"{lab}: {sec.get(k, '').strip()}" for lab, k in zip(labels, keys) if sec.get(k)
                    )
                    return self._strip_markdown(report)
            except Exception as e:
                logger.warning(f"brain template failed, using LLM path: {e}")

        # Build findings list in markdown format
        if not findings:
            findings_str = {
                'ru': 'Патологических изменений не выявлено.',
                'uz': "Patologik o'zgarishlar aniqlanmadi.",
                'en': 'No pathological changes detected.',
            }[language]
            overall_status = {
                'ru': 'Норма',
                'uz': 'Norma',
                'en': 'Normal',
            }[language]
        else:
            findings_str = '\n'.join([
                f"- {f['class_name']}: {f['confidence']*100:.0f}% confidence"
                + (f" ({f.get('location', 'N/A')})" if f.get('location') else '')
                for f in findings
            ])
            high_conf = [f for f in findings if f['confidence'] >= 0.8]
            overall_status = {
                'ru': f"Обнаружены патологии высокой достоверности: {len(high_conf)}" if high_conf else f"Обнаружены находки: {len(findings)}",
                'uz': f"Yuqori ishonchlilikdagi patologiyalar: {len(high_conf)}" if high_conf else f"Topilmalar: {len(findings)}",
                'en': f"High-confidence findings: {len(high_conf)}" if high_conf else f"Findings: {len(findings)}",
            }[language]

        # Patient info is deliberately NOT sent to the LLM (PHI minimisation:
        # the prompt may reach a remote Ollama host or, when explicitly
        # allowed, a cloud API). The template keeps its slots as 'N/A'.
        patient_info = {}

        # Build prompts
        system_prompt = SYSTEM_PROMPTS[language]
        user_prompt = USER_PROMPT_TEMPLATES[language].format(
            modality=modality,
            body_part=body_part,
            patient_id='N/A',
            age='N/A',
            sex='N/A',
            study_date=study_date or 'N/A',
            findings_list=findings_str,
            overall_status=overall_status,
        )

        # Generate via chosen backend
        logger.info(f"Generating {language.upper()} report via {self.backend}...")
        report = ""
        if self.backend == 'ollama':
            report = self._generate_ollama(system_prompt, user_prompt)
        elif self.backend == 'google_ai':
            report = self._generate_google_ai(system_prompt, user_prompt)
        elif self.backend == 'transformers':
            report = self._generate_transformers(system_prompt, user_prompt)

        # Fallback to template if generation failed or template backend
        if not report or self.backend == 'template':
            report = self._fallback_template_report(
                {'findings': findings, 'patient_info': patient_info,
                 'modality': modality, 'body_part': body_part},
                language,
            )

        return self._strip_markdown(report)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown artefacts (##, **, backticks, bullets) so reports
        render cleanly in the plain-text report editor and the exported PDF."""
        if not text:
            return text
        import re
        out = []
        for line in text.splitlines():
            s = re.sub(r'^\s*#{1,6}\s*', '', line)   # '## Heading' -> 'Heading'
            s = s.replace('**', '').replace('__', '').replace('`', '')
            s = re.sub(r'^\s*[-*]\s+', '• ', s)      # normalise bullets
            out.append(s.rstrip())
        cleaned = re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
        return cleaned.strip()

    def _fallback_template_report(self, data: dict, language: str) -> str:
        """Fallback template-based report (when Gemma is not available).

        For brain studies (Modality=MR + BodyPart=BRAIN/HEAD), uses the
        brain-specific templates which cover tumor / hemorrhage / dementia.
        Otherwise uses the generic chest fallback.
        """
        findings = data.get('findings', [])
        modality = (data.get('modality') or '').upper()
        body_part = (data.get('body_part') or '').upper()
        is_brain = (
            'BRAIN' in body_part or 'HEAD' in body_part or
            ('MR' in modality and not body_part)  # MR with no part = assume brain
        )

        # Brain branch — use the rich brain-specific templates
        if is_brain:
            try:
                from src.inference.brain_report_templates import build_brain_report
                report = build_brain_report(findings, language=language, modality=modality)
                section_titles = {
                    'ru': ('КЛИНИЧЕСКОЕ ПОКАЗАНИЕ', 'МЕТОДИКА', 'ОПИСАНИЕ', 'ЗАКЛЮЧЕНИЕ', 'РЕКОМЕНДАЦИИ'),
                    'uz': ("KLINIK KO'RSATMA", 'METODIKA', 'TAVSIF', 'XULOSA', 'TAVSIYALAR'),
                    'en': ('CLINICAL INDICATION', 'TECHNIQUE', 'FINDINGS', 'IMPRESSION', 'RECOMMENDATIONS'),
                }
                titles = section_titles.get(language, section_titles['ru'])
                return (
                    f"{titles[0]}: {report['clinical_indication']}\n\n"
                    f"{titles[1]}: {report['technique']}\n\n"
                    f"{titles[2]}: {report['description']}\n\n"
                    f"{titles[3]}: {report['impression']}\n\n"
                    f"{titles[4]}: {report['recommendation']}"
                )
            except Exception as e:
                logger.debug(f"Brain template fallback failed, using generic: {e}")

        # Generic / chest branch
        if not findings:
            templates = {
                'ru': "КЛИНИЧЕСКОЕ ПОКАЗАНИЕ: Плановое обследование\n\nМЕТОДИКА: Рентгенография в прямой проекции\n\nОПИСАНИЕ: Патологических изменений не выявлено.\n\nЗАКЛЮЧЕНИЕ: Норма.\n\nРЕКОМЕНДАЦИИ: Контроль по клиническим показаниям.",
                'uz': "KLINIK KO'RSATMA: Rejali tekshiruv\n\nMETODIKA: To'g'ri proyeksiyada rentgenografiya\n\nTAVSIF: Patologik o'zgarishlar aniqlanmadi.\n\nXULOSA: Norma.\n\nTAVSIYALAR: Klinik ko'rsatmalarga muvofiq nazorat.",
                'en': "CLINICAL INDICATION: Routine examination\n\nTECHNIQUE: PA chest radiograph\n\nFINDINGS: No pathological changes detected.\n\nIMPRESSION: Normal.\n\nRECOMMENDATIONS: Follow up per clinical indications.",
            }
            return templates[language]

        findings_text = ', '.join([f"{f['class_name']} ({f['confidence']*100:.0f}%)" for f in findings])
        templates = {
            'ru': f"КЛИНИЧЕСКОЕ ПОКАЗАНИЕ: Обследование\n\nМЕТОДИКА: Рентгенография\n\nОПИСАНИЕ: {findings_text}\n\nЗАКЛЮЧЕНИЕ: Рентгенологическая картина соответствует: {findings_text}\n\nРЕКОМЕНДАЦИИ: Консультация специалиста.",
            'uz': f"KLINIK KO'RSATMA: Tekshiruv\n\nMETODIKA: Rentgenografiya\n\nTAVSIF: {findings_text}\n\nXULOSA: Rentgenologik ko'rinish: {findings_text}\n\nTAVSIYALAR: Mutaxassis maslahati.",
            'en': f"CLINICAL INDICATION: Examination\n\nTECHNIQUE: Radiography\n\nFINDINGS: {findings_text}\n\nIMPRESSION: {findings_text}\n\nRECOMMENDATIONS: Specialist consultation.",
        }
        return templates[language]

    # ------------------------------------------------------------
    # DOCTOR Q&A (conversational)
    # ------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        context_findings: list[dict],
        report_text: str,
        language: str = 'ru',
    ) -> str:
        """Answer radiologist's follow-up questions about the AI analysis.

        Example:
            answer_question("Могло ли это быть туберкулезом?", findings, report)
        """
        if self.backend not in ('ollama', 'google_ai'):
            return "Q&A requires Ollama or Google AI backend"

        qa_prompts = {
            'ru': f"""Ты — врач-рентгенолог. Отвечай кратко и по делу на вопросы коллег.

Контекст исследования:
{report_text}

Находки AI: {json.dumps(context_findings, ensure_ascii=False)}

Вопрос коллеги: {question}

Ответь профессионально, используя медицинскую терминологию. Если вопрос выходит за рамки данных, так и скажи.""",

            'uz': f"""Siz rentgenolog shifokorsiz. Hamkasblarning savollariga qisqa va aniq javob bering.

Tekshiruv konteksti:
{report_text}

AI topilmalari: {json.dumps(context_findings, ensure_ascii=False)}

Hamkasbning savoli: {question}

Tibbiy terminologiyadan foydalanib, professional javob bering.""",

            'en': f"""You are a radiologist. Answer colleagues' questions briefly and to the point.

Study context:
{report_text}

AI findings: {json.dumps(context_findings)}

Colleague's question: {question}

Answer professionally using medical terminology.""",
        }

        prompt = qa_prompts.get(language, qa_prompts['ru'])
        if self.backend == 'google_ai':
            return self._generate_google_ai("", prompt)
        return self._generate_ollama("", prompt)
