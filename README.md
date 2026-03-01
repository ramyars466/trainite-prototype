# trainite-prototype
GSoC 2026 prototype - LM Training Toolkit using PyTorch-Ignite
# 🚀 trainite-prototype

A GSoC 2026 prototype for the **trainite** project — a clean, config-driven
LM training toolkit built on **PyTorch + PyTorch-Ignite**.

---

## 📌 Project Goal

Build a lightweight, modular training framework for language models using
PyTorch-Ignite as the core training engine — proposed as part of
**GSoC 2026 (PyTorch-Ignite org)**.

---

## 🧠 Model

- **Architecture:** Decoder-only Transformer
- **Task:** String reversal (`"hello"` → `"olleh"`)
- **Config-driven:** All hyperparameters via YAML/dict config

---

## ✅ Features

- Decoder-only Transformer (PyTorch `nn.TransformerDecoder`)
- PyTorch-Ignite `Engine` for training + validation loops
- `ModelCheckpoint` handler — saves best model (`best.pt`)
- `EarlyStopping` handler
- YAML-style config dictionary (`main.py`, `model.py`, `trainer.py`)
- Clean train/val loss logging per epoch

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Epochs | 80 |
| Final Train Loss | 0.6205 |
| Final Val Loss | 0.9070 |

### Test Results
| Input | Expected | Got | Status |
|-------|----------|-----|--------|
| abcdef | fedcba | fedcba | ✅ |
| ignite | etingi | etingi | ✅ |
| python | nohtyp | nohtyp | ✅ |
| zxywvu | uvwyxz | uvwyxz | ✅ |
| ramyar | raymar | raymra | ❌ |
| hellow | wolleh | wolhew | ❌ |

> ⚠️ Known limitation: repeated characters cause minor prediction errors.
> Next step: augment training data with more repeated-char sequences.




