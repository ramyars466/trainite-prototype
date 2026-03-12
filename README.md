# 🚀 Trainite Prototype

### GSoC 2026 Prototype — Modular LM Training Toolkit built with PyTorch-Ignite

This repository contains a **prototype implementation of Trainite**, a modular training toolkit for language models built on top of **PyTorch** and **PyTorch-Ignite**.

The goal of this prototype is to demonstrate how a **clean, extensible training framework** can be designed for research workflows while minimizing boilerplate code.

This project explores how **PyTorch-Ignite can serve as the core training engine** for a lightweight but powerful ML training toolkit.

---

# 📌 Project Motivation

Training deep learning models typically requires writing repetitive code for:

* training loops
* validation logic
* checkpointing
* experiment tracking
* dataset and model management

The goal of **Trainite** is to provide a **lightweight, modular training framework** that simplifies these tasks while remaining flexible for research and experimentation.

This prototype explores a design where **datasets, models, and training pipelines are dynamically configurable and extensible**.

---

# 🧠 Key Idea

Trainite follows a **config-driven architecture** where experiments are defined using a configuration file.

The framework automatically handles:

* dataset loading
* model creation
* training loops
* validation
* checkpointing
* experiment tracking

All operations are controlled through a **simple CLI interface**.

---

# ✨ Implemented Features

## CLI Interface

Train models and run inference directly from the command line.

```bash
trainite train
trainite generate hello
```

---

## Config-Driven Experiments

All experiment parameters are defined in:

```
trainite/configs/config.yaml
```

Example:

```yaml
dataset:
  name: reverse
  params:
    vocab: "abcdefghijklmnopqrstuvwxyz"
    seq_length: 10
    dataset_size: 20000

model:
  name: transformer
  params:
    embed_dim: 128
    num_heads: 4
    num_layers: 3
    dropout: 0.1

training:
  batch_size: 64
  lr: 0.0003
  max_epochs: 30
  output_dir: output
```

This allows experiments to be **fully reproducible**.

---

# 🧩 Dataset Registry System

Datasets are dynamically registered using a **registry pattern**.

Example:

```python
@register_dataset("reverse")
class StringReversalDataset(Dataset):
```

Datasets can be loaded dynamically:

```python
dataset = get_dataset("reverse")
```

Trainite also supports **plugin datasets**:

```
trainite register-dataset my_dataset.py
```

---

# 🧩 Model Registry System

Models are registered dynamically and selected via configuration.

Supported architectures in this prototype:

* Transformer
* LSTM
* GRU

Example:

```yaml
model:
  name: transformer
```

External models can be added using:

```
trainite register-model my_model.py
```

---

# ⚙️ Training Engine

Training is implemented using **PyTorch-Ignite Engine**, providing:

* modular training loops
* validation loops
* checkpoint saving
* early stopping
* TensorBoard logging

Implemented Ignite handlers include:

* `ModelCheckpoint`
* `EarlyStopping`
* TensorBoard logging

---

# 📊 Experiment Tracking

Each training run automatically creates an experiment folder:

```
experiments/
    run_20260312_210824/
        config.json
        metrics.json
```

This allows experiments to be:

* reproducible
* comparable
* organized

---

# 📈 Experiment Viewer

List all experiment runs:

```
trainite experiments
```

Example output:

```
run_20260307_141805
run_20260308_151124
run_20260312_210824
```

---

# 🔎 Experiment Inspection

View the details of a specific experiment:

```
trainite experiment run_20260312_210824
```

Displays:

* experiment configuration
* training metrics

---

# 📊 Experiment Comparison

Compare model performance across experiment runs:

```
trainite compare
```

Example output:

```
Run                          Best Val Loss
---------------------------------------------
run_20260307_161011          0.000733
run_20260312_210824          0.000812
run_20260308_151124          0.001112
```

This helps quickly identify the **best performing experiment**.

---

# 🧪 Example Task

The prototype demonstrates training models to perform **sequence reversal**.

Example:

```
Input  : hello
Output : olleh
```

This toy task verifies the correctness of:

* dataset pipeline
* training pipeline
* inference pipeline

---

# 📊 Training Results

Example training output:

| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 1     | 2.31       | 2.02     |
| 5     | 0.60       | 0.36     |
| 10    | 0.17       | 0.03     |
| 20    | 0.05       | 0.01     |

---

# 🏗 Project Architecture

```
trainite/
│
├── cli.py
├── train.py
│
├── configs/
│   └── config.yaml
│
├── datasets/
│   ├── registry.py
│   └── string_reverse.py
│
├── models/
│   ├── registry.py
│   ├── transformer.py
│   ├── lstm.py
│   └── gru.py
│
├── trainers/
│   └── ignite_trainer.py
│
├── utils/
│   ├── inference.py
│   ├── experiment.py
│   └── experiment_viewer.py
│
└── plugins/
    ├── datasets.json
    └── models.json
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/ramyars466/trainite-prototype
cd trainite-prototype
```

Install the package:

```
pip install -e .
```

---

# 🚀 CLI Commands

| Command                     | Description                |
| --------------------------- | -------------------------- |
| `trainite train`            | Train model                |
| `trainite generate <text>`  | Run inference              |
| `trainite experiments`      | List experiment runs       |
| `trainite experiment <run>` | Show experiment details    |
| `trainite compare`          | Compare experiment results |
| `trainite register-dataset` | Register dataset plugin    |
| `trainite register-model`   | Register model plugin      |

---

# 📈 TensorBoard Visualization

Start TensorBoard:

```
tensorboard --logdir runs
```

Open:

```
http://localhost:6006
```

to visualize training curves.

---

# 🧩 Design Principles

Trainite is designed around:

* modular architecture
* registry-based extensibility
* config-driven experiments
* minimal training boilerplate
* extensible research workflows

---

# 🔮 Future Improvements

Potential future extensions include:

* experiment dashboard UI
* dataset plugin ecosystem
* hyperparameter search support
* distributed training
* multi-task training
* model evaluation tools

---

# 🎓 GSoC Context

This prototype was developed as a **proof-of-concept for the Trainite project idea** proposed under **Google Summer of Code 2026 with the PyTorch-Ignite organization**.

The goal is to design a **lightweight but extensible training toolkit for language models built around PyTorch-Ignite's flexible engine system**.

---

# 🙌 Acknowledgements

* PyTorch
* PyTorch-Ignite
* Google Summer of Code
  
<img width="769" height="456" alt="image" src="https://github.com/user-attachments/assets/7c45d2b7-9837-4e48-b505-e9143fa4a033" />
<img width="626" height="130" alt="image" src="https://github.com/user-attachments/assets/dee03c5a-3531-4026-934e-2c383f238a30" />
<img width="643" height="308" alt="image" src="https://github.com/user-attachments/assets/e1daf942-8ce1-4a59-96b3-2efa2aa465ea" />




