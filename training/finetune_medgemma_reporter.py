#!/usr/bin/env python3
"""
================================================================================
  SENTINEL — MEDGEMMA REPORTER FINE-TUNING (LoRA)
================================================================================

  Fine-tunes Google MedGemma 4B-IT on radiology reports from your clinic so
  it learns:
    • Russian/Uzbek medical phrasing your doctors actually use
    • Local report structure conventions
    • Pathology terminology specific to brain MRI

  Uses LoRA (Low-Rank Adaptation) so only ~1 % of parameters change. This
  means:
    • Trains in ~30 minutes on a single RTX 3060 / 4070
    • Output is a small adapter file (~50 MB), not a full 8 GB model
    • Original weights stay intact — adapter swap is reversible

  Data format expected (JSONL):
    data/reports.jsonl
      {"image_path": "scan_001.png", "findings": "Pneumonia 87% right lower",
       "report": "ПРОТОКОЛ...\\nЗАКЛЮЧЕНИЕ: правосторонняя пневмония..."}

  Run:
    python training/finetune_medgemma_reporter.py \\
        --data data/reports.jsonl \\
        --base-model unsloth/medgemma-4b-it \\
        --epochs 3 \\
        --output models/medgemma_uzbek_lora

  Loaded automatically by the inference server when models/medgemma_uzbek_lora/
  exists. The default Gemma generation path adds the LoRA on top.
================================================================================
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Make repo importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True,
                     help='JSONL with {image_path, findings, report} per line')
    p.add_argument('--base-model', default='unsloth/medgemma-4b-it')
    p.add_argument('--output', type=Path, default=Path('models/medgemma_uzbek_lora'))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--max-seq-length', type=int, default=2048)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = p.parse_args()

    print("=" * 70)
    print("  MedGemma Reporter LoRA Fine-Tuning")
    print("=" * 70)

    # Lazy imports — these are heavy
    try:
        import torch
        from transformers import (
            AutoProcessor, AutoModelForImageTextToText,
            TrainingArguments, Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("  Install: pip install transformers peft accelerate bitsandbytes")
        sys.exit(1)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else \
                 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"[device] {device}")

    if device == 'cpu':
        print("⚠ Training MedGemma on CPU is impractically slow. Use a GPU "
              "(local CUDA, Apple Silicon MPS, or rent an RTX 4090 on RunPod).")

    # Load base model
    print(f"[model] loading {args.base_model} (this may download ~8 GB on first run)…")
    dtype = torch.float16 if device != 'cpu' else torch.float32
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device if device != 'cpu' else 'cpu',
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable:,} / total {total:,} ({100*trainable/total:.2f}%)")

    # Load data
    print(f"[data] reading {args.data}")
    samples = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    print(f"[data] {len(samples)} samples")

    if not samples:
        print("✗ No training samples found.")
        sys.exit(1)

    # Build dataset
    from torch.utils.data import Dataset

    class ReportDataset(Dataset):
        def __init__(self, items, processor, max_len):
            self.items = items
            self.processor = processor
            self.max_len = max_len

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            item = self.items[i]
            from PIL import Image
            img_path = Path(item['image_path'])
            if not img_path.is_absolute():
                img_path = args.data.parent / img_path
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                # Synthetic placeholder if image missing — text-only fine-tune
                img = Image.new('RGB', (224, 224), color=128)

            prompt = (
                "You are a radiologist. Given this brain MRI scan and the AI findings, "
                "write a structured radiology report in Russian.\n\n"
                f"AI findings: {item.get('findings', '')}\n\n"
                "Report:\n"
            )
            target = item['report']

            full_text = prompt + target
            inputs = self.processor(
                text=full_text,
                images=img,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['labels'] = inputs['input_ids'].clone()
            return inputs

    train_ds = ReportDataset(samples, processor, args.max_seq_length)

    # Train
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=20,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        fp16=(device == 'cuda'),
        bf16=(device == 'mps'),
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    print("[train] starting…")
    trainer.train()

    # Save LoRA adapter only (small)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)

    # Save manifest
    manifest = {
        'base_model': args.base_model,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'trained_samples': len(samples),
        'epochs': args.epochs,
    }
    (args.output / 'sentinel_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f"\n✓ DONE. Adapter saved to {args.output}/")
    print(f"  Adapter size: {sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file()) // (1024*1024)} MB")
    print()
    print("  To use it in the inference server, set:")
    print(f"    export MEDGEMMA_LORA={args.output.resolve()}")
    print("  Then restart the server.")


if __name__ == '__main__':
    main()
