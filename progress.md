# Trainite Prototype - Project Progress & Handoff Document

## Project Overview
**Trainite** is a modular toolbox for training language models built on top of **PyTorch-Ignite**. It is the prototype implementation for the official PyTorch-Ignite Google Summer of Code 2026 project: *"Development of a toolbox for training language models."*

The goal is to provide researchers with a reusable, configuration-driven codebase where they can easily plug in custom models and datasets without having to rewrite boilerplate training loops, evaluation loops, or checkpointing logic.

## Architecture Highlights
- **CLI Layer (Typer):** Commands like `trainite train`, `trainite init`, `trainite experiments`, and `trainite compare` drive the workflow.
- **Configuration Layer (YAML):** All hyperparameters, dataset configurations, and model definitions are handled through YAML, making experiments 100% reproducible.
- **Registry System:** Uses decorators (`@register_model`, `@register_dataset`) and dynamic module loading (`importlib`) to inject custom code at runtime without modifying the core engine.
- **Ignite Engine:** The core training loop is built using PyTorch-Ignite `Engine`, `Events`, and `Handlers` (EarlyStopping, Checkpointing).
- **Experiment Tracking:** Every run automatically creates a timestamped output folder containing the saved model (`best.pt`, `last.pt`), `metrics.json`, the exact `config.yaml` used, and a `code/` snapshot directory.

---

## 🟢 What Has Been Completed (Current State)
The core foundation of the prototype is fully functional and aligned with the GSoC idea requirements:
1. **Dynamic Plugin Registries:** The registry system can dynamically load built-in models/datasets AND auto-load custom `models/*.py` and `datasets/*.py` files from the user's local working directory.
2. **Experiment Tracking & Code Snapshots:** Training runs automatically generate reproducible output folders containing `metrics.json`, `config.json`, and an exact copy of the code (`main.py`, `config.yaml`, local plugins) that was used to run the experiment.
3. **Exact Checkpointing:** PyTorch-Ignite Handlers have been customized to save exactly `best.pt` and `last.pt` to cleanly match the GSoC requirements.
4. **Scaffolding (`trainite init`):** The `init` command automatically generates a clean workspace (`configs/config.yaml`, `models/lang_model.py`, `datasets/str_rev_dataset.py`, and `main.py`) containing working dummy templates with random integer generation so that out-of-the-box training works without throwing `NaN` losses.
5. **Experiment Viewer:** CLI commands `trainite experiments` and `trainite compare` successfully read and list historical metrics from the output directories.

---

## 🟡 What Needs to Be Done Next (GSoC Roadmap)
If you are an AI agent picking up this project, here are the next phases of development based on the GSoC proposal timeline:

### 1. The Evaluation Workflow (High Priority)
- **Task:** Implement `trainite evaluate`.
- **Details:** The training loop works great, but evaluation needs to be a first-class CLI command. A user should be able to run `trainite evaluate --checkpoint=experiments/run_X/best.pt` and get validation metrics on a test set without writing a custom evaluation script.

### 2. TensorBoard Integration
- **Task:** Attach PyTorch-Ignite's TensorBoard Logger.
- **Details:** While metrics are currently saved to `metrics.json` and printed to the console, the toolbox needs to utilize Ignite's `TensorboardLogger` handler to write logs so users can visualize training curves dynamically.

### 3. Model Architecture Expansion
- **Task:** Expand the built-in models beyond the simple `DecoderOnlyTransformer` and `GRU`.
- **Details:** Ensure the registry and configuration system cleanly supports more complex architectures (like Mamba or RWKV as mentioned in the GSoC idea) and that positional encodings and masking behave correctly across them.

### 4. Packaging and Documentation
- **Task:** Prepare the project for PyPI distribution.
- **Details:** The codebase needs formal PyTorch-Ignite style docstrings, a comprehensive `README.md`, and a robust `setup.py`/`pyproject.toml` so a user can genuinely run `pip install trainite`.

---
*Note to future AI agents: Always prioritize using the dynamic registry system for adding new features. Do not hardcode new models or datasets into the `train.py` engine directly. Keep the core engine agnostic!*
