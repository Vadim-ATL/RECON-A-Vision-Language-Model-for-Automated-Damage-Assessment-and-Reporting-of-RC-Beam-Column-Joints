# RECON

Vision-language model for automated damage assessment and report generation on reinforced concrete (RC) beam–column joints.
Given a joint image, RECON produces a structured engineering report covering joint type, damage types, failure mechanism, severity (joint + beam), and a free-text analysis.

---

## What it does

RECON has proven high precision and accurate in reinforced concrete beam-column structure report generation, the model was evaluated on two kinds of metrics:

---

## Architecture

Modular BLIP-2–style stack (~3.5B params loaded, ~500M trained): Image → FieldAwareMoE (DINOv2 + field MoE) → Q-Former (BLIP-2 + aux classification heads) → LLMProjector → LLaMA-3.2-3B (mostly frozen) → Structured report

| Module | File | Role |
|--------|------|------|
| Vision encoder | `dinov2_vision_encoder.py` | DINOv2 + per-field MoE → 261 tokens |
| Bridge | `qformer_bridge.py` | BLIP-2 Q-Former + form heads → 32 tokens |
| VLM wrapper | `baseline_vlm.py` | Fuses visual tokens into LLaMA |
| Training | `train_full_vlm.py` | SCST fine-tuning + aux losses |

**Field-aware MoE** routes vision through 4 experts per semantic field: joint, damage, location, severity_j, severity_b.

**Aux heads** on Q-Former tokens 0–8 supervise joint type, failure mode, severities, and multi-label damage — this is the FPM path.

---

## Project layout
RECON/ 
├── dinov2_vision_encoder.py # FieldAwareMoE 
├── qformer_bridge.py # QFormer + LLMProjector 
├── baseline_vlm.py # main model 
├── train_full_vlm.py # Training 
├── inference_vlm.py # Inference
├── evaluate.py # NLGM metrics (pycocoevalcap) 

---

## Requirements
- Python 3.10+
- CUDA GPU (tested on single RTX 5090)
- Hugging Face access for:
  - `facebook/dinov2-base`
  - `Salesforce/blip2-opt-2.7b`
  - `meta-llama/Llama-3.2-3B-Instruct`
    
```bash
pip install torch torchvision transformers pandas pillow tqdm nltk
pip install pycocoevalcap bert-score rouge-score tabulate
