# SimKGC Persian + English Knowledge Graph Pipeline

A lightweight Knowledge Graph Completion (KGC) pipeline using **SimKGC**, **Vocabulary Pruning (FA + EN)**, **Matryoshka 256-d Representation Learning**, and **INT8 Quantization** for high-speed edge inference in Centrode (Rust/Flutter).

---

## Architecture Overview

1. **Dual Encoder:**
   * Head Encoder: $\text{Head} + \text{" [SEP] "} + \text{Relation} \longrightarrow \mathbf{u} \in \mathbb{R}^{256}$
   * Tail Encoder: $\text{Tail} \longrightarrow \mathbf{v} \in \mathbb{R}^{256}$
2. **Vocabulary Pruning:** Slices off unused scripts, keeping strictly Persian (`\u0600-\u06FF`) and English (`a-zA-Z`), reducing embedding table size by ~60%.
3. **Matryoshka 256d + 128d:** Multi-dimensional InfoNCE contrastive loss allows high-speed dot products on 256-d normalized vectors.
4. **Rust Binary Serialization:** Exports the candidate concept database into a zero-copy INT8 binary format (`concepts_256d_int8.bin`).

---

## Directory Structure

```text
simkgc_pipeline/
├── .venv/                         # Local Python virtual environment
├── requirements.txt               # Dependencies for Kaggle/local training
├── config/
│   └── training_config.yaml       # Hyperparameters & paths
├── data/
│   ├── raw/
│   │   └── conceptnet_sample.json # Sample Persian + English ConceptNet assertions
│   └── synthetic/
│       └── mock_triples.json      # Sample generated concept triples
├── src/
│   ├── data/
│   │   ├── conceptnet_extractor.py # ConceptNet sampler/streamer
│   │   └── mock_llm_generator.py   # Synthetic triplet generator (Qwen2.5 / Mock)
│   ├── model/
│   │   ├── biencoder.py            # Dual transformer with Matryoshka head
│   │   ├── vocab_pruner.py         # Vocabulary trimmer for FA + EN
│   │   └── loss.py                 # InfoNCE + in-batch negative loss
│   ├── train.py                    # Training entrypoint
│   └── export.py                   # ONNX exporter, INT8 quantizer, and Rust binary writer
└── exports/
    ├── concepts_256d_int8.bin      # Sample INT8 flat binary matrix for Rust
    └── concepts_dict.json          # Index-to-string mapping
```

---

## Running on Kaggle GPUs (2x T4)

### 1. Upload this folder to a Kaggle Notebook
1. Enable **GPU T4 x 2** in Kaggle settings.
2. Install requirements in the Kaggle notebook cell:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Generate / Augment Synthetic Triples
Run Qwen 2.5 on Kaggle to generate rich Persian commonsense and scientific triplets:
```python
from src.data.mock_llm_generator import generate_mock_synthetic_triples
generate_mock_synthetic_triples(Path("data/synthetic/triples.json"))
```

### 3. Run Training
```bash
python src/train.py
```

### 4. Export ONNX and Rust Binary Assets
```bash
python src/export.py
```
Download the resulting `exports/simkgc_256d_int8.onnx` and `exports/concepts_256d_int8.bin` into Centrode's assets directory.
