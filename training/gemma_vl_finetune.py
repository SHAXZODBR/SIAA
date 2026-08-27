"""
================================================================================
  SENTINEL MEDICAL AI — GEMMA 3 VISION-LANGUAGE FINE-TUNING
================================================================================
  Fine-tunes Google's Gemma 3 multimodal model to generate radiology reports
  from chest X-ray images + DenseNet121 findings.

  ARCHITECTURE:
    Chest X-ray (image) + DenseNet121 findings (text)
                         ↓
                    Gemma 3 VL
                         ↓
    Natural Russian/Uzbek/English radiology report

  USE CASES:
    1. Generate draft report text from image + findings
    2. Answer radiologist questions about the image
    3. Translate findings to patient-friendly language
    4. Explain complex pathologies in natural language

  WHY GEMMA 3 vs Qwen2-VL:
    - Better Russian/Uzbek language understanding
    - Google's multilingual training
    - Smaller, more efficient variants available (4B vs 7B+)
    - Better alignment for structured output

  TRAINING DATA NEEDED:
    Pairs of (X-ray image, radiologist report text) — 1000+ pairs ideal

  PLATFORMS TO RUN:
    - Kaggle (free T4 x2) - sufficient for Gemma 3 4B with QLoRA
    - Colab Pro ($10/mo) - T4/V100
    - RunPod (paid) - A40/A100 for 27B variant

  FINE-TUNING METHODS:
    - LoRA (full fine-tune) — smaller Gemma 4B
    - QLoRA (4-bit quantized) — run 27B on free GPU

  HOW TO USE:
    # 1. Install deps (Kaggle/Colab):
    !pip install -q transformers peft bitsandbytes accelerate trl datasets

    # 2. Run this script
    python gemma_vl_finetune.py
================================================================================
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ===================================================================
# CONFIG
# ===================================================================

CONFIG = {
    # Base model — pick based on your GPU:
    #   'google/gemma-3-4b-it'        : 4B params, fits on T4 (16GB)
    #   'google/gemma-3-12b-it'       : 12B params, needs A40 (48GB) or QLoRA
    #   'google/gemma-3-27b-it'       : 27B params, needs A100 (80GB) or QLoRA
    'base_model': 'google/gemma-3-4b-it',

    # Training data
    'dataset_path': '/kaggle/working/radiology_reports.jsonl',  # or your path
    'output_dir': '/kaggle/working/gemma_medical',

    # LoRA config (efficient fine-tuning)
    'use_qlora': True,              # 4-bit quantization — fits on 16GB GPU
    'lora_r': 16,                    # LoRA rank
    'lora_alpha': 32,                # LoRA alpha (usually 2x rank)
    'lora_dropout': 0.05,
    'target_modules': [              # Which layers to fine-tune
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],

    # Training hyperparameters
    'epochs': 3,
    'batch_size': 2,                 # Small for multimodal (images are big)
    'gradient_accumulation': 8,      # Effective batch 16
    'learning_rate': 2e-4,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'max_seq_length': 1024,          # Max tokens in prompt+response
    'image_size': 448,               # Gemma 3 supports 448x448 images

    # Generation settings for inference
    'max_new_tokens': 512,
    'temperature': 0.3,              # Low for medical accuracy
    'top_p': 0.9,
    'repetition_penalty': 1.1,

    # Hardware
    'mixed_precision': 'bf16',       # bfloat16 for stability
    'seed': 42,

    # Language
    'primary_language': 'ru',        # Russian primary
    'supported_languages': ['ru', 'uz', 'en'],
}


# ===================================================================
# DATASET FORMAT
# ===================================================================

EXAMPLE_DATASET_FORMAT = """
# Your dataset should be JSONL with this format:

{"image_path": "/kaggle/input/data/images_001/images/00000001_000.png",
 "findings_text": "DenseNet findings: Pneumonia 87%, Effusion 62% (Right lower zone)",
 "report": "Klinicheskoye pokazaniye: ...", "language": "ru"}

{"image_path": "/kaggle/input/data/images_001/images/00000002_000.png",
 "findings_text": "DenseNet findings: Cardiomegaly 73%",
 "report": "Klinik ko'rsatma: ...", "language": "uz"}

# Minimum 500-1000 examples for good fine-tuning.
# Ideal: 5000-10000 examples covering all pathologies.
"""


# ===================================================================
# PROMPT TEMPLATES — Medical Report Generation
# ===================================================================

SYSTEM_PROMPT_RU = """Ты — опытный врач-рентгенолог. На основе рентгеновского снимка
и результатов ИИ-анализа составь структурированный протокол рентгенологического
исследования. Используй стандартную медицинскую терминологию.

Структура протокола:
1. КЛИНИЧЕСКОЕ ПОКАЗАНИЕ
2. МЕТОДИКА
3. ОПИСАНИЕ (подробное описание находок)
4. ЗАКЛЮЧЕНИЕ (краткий диагностический вывод)
5. РЕКОМЕНДАЦИИ (дальнейшие действия)

Будь точным, объективным, избегай домыслов."""

SYSTEM_PROMPT_UZ = """Siz tajribali rentgenolog shifokorsiz. Rentgen suratga va
AI-tahlil natijalariga asoslanib, rentgenologik tekshiruv bayonnomasini
tuzing. Standart tibbiy terminologiyadan foydalaning.

Bayonnoma tuzilishi:
1. KLINIK KO'RSATMA
2. METODIKA
3. TAVSIF
4. XULOSA
5. TAVSIYALAR

Aniq, ob'ektiv bo'ling, taxminlardan qoching."""

SYSTEM_PROMPT_EN = """You are an experienced radiologist. Based on the X-ray
image and AI analysis findings, write a structured radiology report using
standard medical terminology.

Report structure:
1. CLINICAL INDICATION
2. TECHNIQUE
3. FINDINGS (detailed description)
4. IMPRESSION (diagnostic conclusion)
5. RECOMMENDATIONS (next steps)

Be precise, objective, and avoid speculation."""

SYSTEM_PROMPTS = {
    'ru': SYSTEM_PROMPT_RU,
    'uz': SYSTEM_PROMPT_UZ,
    'en': SYSTEM_PROMPT_EN,
}


def build_prompt(findings_text: str, language: str = 'ru') -> str:
    """Build the user prompt for Gemma."""
    system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_RU)

    if language == 'ru':
        user_prompt = f"""Результаты ИИ-анализа рентгеновского снимка:
{findings_text}

На основе снимка и этих данных составь полный рентгенологический протокол."""
    elif language == 'uz':
        user_prompt = f"""Rentgen suratning AI-tahlil natijalari:
{findings_text}

Suratga va ushbu ma'lumotlarga asoslanib, to'liq rentgenologik bayonnoma tuzing."""
    else:
        user_prompt = f"""AI analysis results from the X-ray:
{findings_text}

Based on the image and these findings, write a full radiology report."""

    return system, user_prompt


# ===================================================================
# DATA PREPARATION
# ===================================================================

def prepare_dataset(dataset_path: str, processor):
    """Load and tokenize the radiology dataset for Gemma fine-tuning."""
    from datasets import load_dataset
    from PIL import Image

    print(f"[DATA] Loading dataset from {dataset_path}...")

    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset not found at {dataset_path}")
        print(EXAMPLE_DATASET_FORMAT)
        return None

    # Load JSONL
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    print(f"  Loaded {len(dataset)} examples")

    def format_example(example):
        """Format each (image, findings, report) tuple into Gemma chat format."""
        image = Image.open(example['image_path']).convert('RGB')
        language = example.get('language', 'ru')
        system_prompt, user_prompt = build_prompt(example['findings_text'], language)

        # Gemma 3 chat template with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example['report']}],
            },
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=CONFIG['max_seq_length'],
            truncation=True,
        )

        # Labels = input_ids (causal LM)
        inputs['labels'] = inputs['input_ids'].clone()
        # Mask padding tokens from loss
        inputs['labels'][inputs['labels'] == processor.tokenizer.pad_token_id] = -100

        return {k: v.squeeze(0) for k, v in inputs.items()}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return dataset


# ===================================================================
# MODEL LOADING — Gemma 3 with QLoRA
# ===================================================================

def load_model_and_processor():
    """Load Gemma 3 with 4-bit quantization and LoRA adapters."""
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers import Gemma3ForConditionalGeneration

    print(f"[MODEL] Loading {CONFIG['base_model']}...")

    # QLoRA — 4-bit quantization saves massive VRAM
    if CONFIG['use_qlora']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        CONFIG['base_model'],
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',  # safer with quantization
    )

    # Processor handles both text and images
    processor = AutoProcessor.from_pretrained(CONFIG['base_model'])

    # Prepare for kbit training (if using QLoRA)
    if CONFIG['use_qlora']:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply LoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ===================================================================
# TRAINING LOOP
# ===================================================================

def train():
    """Main fine-tuning function."""
    from transformers import TrainingArguments, Trainer

    # Load model + processor
    model, processor = load_model_and_processor()

    # Load dataset
    dataset = prepare_dataset(CONFIG['dataset_path'], processor)
    if dataset is None:
        return

    # Split train/val (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=CONFIG['seed'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation'],
        learning_rate=CONFIG['learning_rate'],
        warmup_ratio=CONFIG['warmup_ratio'],
        weight_decay=CONFIG['weight_decay'],
        max_grad_norm=CONFIG['max_grad_norm'],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        eval_strategy='steps',
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        seed=CONFIG['seed'],
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=processor.tokenizer,
    )

    # Train
    print("\n[TRAIN] Starting fine-tuning...")
    trainer.train()

    # Save final model
    print(f"\n[SAVE] Saving model to {CONFIG['output_dir']}")
    trainer.save_model(CONFIG['output_dir'])
    processor.save_pretrained(CONFIG['output_dir'])

    print("\n[DONE] Fine-tuning complete!")
    print(f"  LoRA adapter saved to: {CONFIG['output_dir']}")
    print(f"  Download this folder to deploy locally")

    return model, processor


# ===================================================================
# INFERENCE — Generate reports from trained model
# ===================================================================

def generate_report(
    image_path: str,
    findings_text: str,
    language: str = 'ru',
    model=None,
    processor=None,
    checkpoint_path: Optional[str] = None,
) -> str:
    """Generate a radiology report from image + DenseNet findings."""
    from PIL import Image
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    if model is None or processor is None:
        print(f"[LOAD] Loading fine-tuned model from {checkpoint_path or CONFIG['output_dir']}")

        # Load base model
        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            CONFIG['base_model'],
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path or CONFIG['output_dir'])
        model.eval()

        processor = AutoProcessor.from_pretrained(checkpoint_path or CONFIG['output_dir'])

    # Build prompt
    image = Image.open(image_path).convert('RGB')
    system_prompt, user_prompt = build_prompt(findings_text, language)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
            ],
        },
    ]

    # Tokenize
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG['max_new_tokens'],
            temperature=CONFIG['temperature'],
            top_p=CONFIG['top_p'],
            repetition_penalty=CONFIG['repetition_penalty'],
            do_sample=True,
        )

    # Decode only the generated part
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    report_text = processor.decode(generated, skip_special_tokens=True)
    return report_text.strip()


# ===================================================================
# SAMPLE DATASET CREATION (for testing without real clinic data)
# ===================================================================

def create_synthetic_dataset(output_path: str, num_examples: int = 1000):
    """Create a synthetic dataset from NIH + example reports.

    For REAL training, you need actual radiologist reports from clinics.
    This is just a starting template.
    """
    print(f"[SYNTHETIC] Creating {num_examples} synthetic examples...")

    # Sample reports per pathology in Russian
    sample_reports_ru = {
        'Pneumonia': "КЛИНИЧЕСКОЕ ПОКАЗАНИЕ:\nОбследование органов грудной клетки.\n\nМЕТОДИКА:\nРентгенография в прямой проекции.\n\nОПИСАНИЕ:\nВ правой нижней доле определяется участок инфильтрации легочной ткани средней интенсивности с нечеткими контурами. Корень легкого структурный. Синус затемнен.\n\nЗАКЛЮЧЕНИЕ:\nРентгенологическая картина соответствует правосторонней нижнедолевой пневмонии.\n\nРЕКОМЕНДАЦИИ:\nКонсультация пульмонолога. Контроль через 10-14 дней.",
        'Cardiomegaly': "КЛИНИЧЕСКОЕ ПОКАЗАНИЕ:\nОбследование сердца.\n\nМЕТОДИКА:\nРентгенография грудной клетки.\n\nОПИСАНИЕ:\nСердечная тень расширена. Кардиоторакальный индекс увеличен. Легочные поля без очаговых изменений.\n\nЗАКЛЮЧЕНИЕ:\nКардиомегалия.\n\nРЕКОМЕНДАЦИИ:\nЭхокардиография. Консультация кардиолога.",
        'Effusion': "КЛИНИЧЕСКОЕ ПОКАЗАНИЕ:\nОбследование органов грудной клетки.\n\nМЕТОДИКА:\nРентгенография.\n\nОПИСАНИЕ:\nВ правой плевральной полости определяется наличие жидкости до уровня VI ребра. Синус затемнен.\n\nЗАКЛЮЧЕНИЕ:\nПравосторонний плевральный выпот.\n\nРЕКОМЕНДАЦИИ:\nУЗИ плевральных полостей. Консультация терапевта.",
        'Normal': "КЛИНИЧЕСКОЕ ПОКАЗАНИЕ:\nПрофилактическое обследование.\n\nМЕТОДИКА:\nРентгенография грудной клетки в прямой проекции.\n\nОПИСАНИЕ:\nЛегочные поля прозрачны, без очаговых и инфильтративных изменений. Корни легких структурны. Сердечная тень обычной формы. Синусы свободны.\n\nЗАКЛЮЧЕНИЕ:\nПатологических изменений не выявлено.\n\nРЕКОМЕНДАЦИИ:\nКонтроль согласно плану.",
    }

    with open(output_path, 'w') as f:
        for i in range(num_examples):
            # Random pathology
            import random
            pathology = random.choice(list(sample_reports_ru.keys()))
            confidence = random.randint(60, 95)

            findings = f"DenseNet121 findings: {pathology} {confidence}%"
            if pathology == 'Normal':
                findings = "DenseNet121 findings: Normal (no pathology detected)"

            example = {
                'image_path': f"/kaggle/input/data/images_001/images/{i:08d}_000.png",
                'findings_text': findings,
                'report': sample_reports_ru[pathology],
                'language': 'ru',
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"  Saved {num_examples} examples to {output_path}")
    print(f"\n  NOTE: This is synthetic. For production, use REAL radiologist reports from clinics.")


# ===================================================================
# MAIN
# ===================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'generate', 'create_synthetic'],
                        default='train', help='Mode to run')
    parser.add_argument('--image', type=str, help='Image path for generation')
    parser.add_argument('--findings', type=str, help='DenseNet findings text')
    parser.add_argument('--language', default='ru', choices=['ru', 'uz', 'en'])
    parser.add_argument('--checkpoint', type=str, help='Fine-tuned model path')
    parser.add_argument('--dataset_out', default='radiology_reports.jsonl')
    parser.add_argument('--num_synthetic', type=int, default=1000)
    args = parser.parse_args()

    if args.mode == 'create_synthetic':
        create_synthetic_dataset(args.dataset_out, args.num_synthetic)
    elif args.mode == 'train':
        train()
    elif args.mode == 'generate':
        if not args.image or not args.findings:
            print("Provide --image and --findings")
        else:
            report = generate_report(
                args.image, args.findings, args.language,
                checkpoint_path=args.checkpoint,
            )
            print("=" * 70)
            print("GENERATED REPORT:")
            print("=" * 70)
            print(report)
