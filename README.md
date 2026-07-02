# SPA-SNN: Bio-Plausible Spiking Neural Network for Action Recognition

A fully bio-plausible Spiking Neural Network (SNN) pipeline for real-time human action recognition using Dynamic Vision Sensor (DVS) event cameras. The system achieves **89.2% validation accuracy** on the DailyAction-DVS 12-class benchmark using a **Leave-One-Subject-Out (LOSO)** cross-validation split — with **no global backpropagation**.

---

## Key Design Principles

This project strictly enforces biological plausibility at every layer:

| Constraint | Implementation |
|---|---|
| No convolutions | Local Receptive Fields with spatial masks |
| No BPTT / global backprop | STDP (L1) + Deep E-Prop (L2/L3) |
| Dale's Law | W ≥ 0 enforced in L1 (excitatory-only synapses) |
| Sparse activity | ~2% average spike rate (ultra-low energy) |
| Anti-overfitting | Saccadic augmentation (microsaccade simulation) |
| Temporal fidelity | Asynchronous event streams, 50ms bins, 2s window |

---

## Architecture

```
DVS Event Stream (128×128, 2 polarities)
         │
         ▼
┌─────────────────────────────────────┐
│  Layer 1 — Local Receptive Fields   │  ← STDP unsupervised (Bichler 2012)
│  8,192 LIF neurons                  │    Gabor/DoG initialization
│  Multi-scale radii: 6, 10, 16 px   │    Dale's Law (W ≥ 0)
│  Adaptive threshold homeostasis     │    Noise suppression (Weight Decay)
└─────────────────────────────────────┘
         │  spk1 (sparse, ~2%)
         ▼
┌─────────────────────────────────────┐
│  Layer 2 — E-Prop Hidden 1          │  ← Deep E-Prop (Bellec et al. 2020)
│  4,096 ReLU units                   │    Local eligibility traces
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Layer 3 — E-Prop Hidden 2          │
│  1,024 ReLU units                   │
└─────────────────────────────────────┘
         │
         ▼
    12-class output
```

---

## Results — DailyAction-DVS LOSO Benchmark

**Dataset**: DailyAction-DVS — 1,440 recordings, 12 action classes, DVS128 sensor (128×128)  
**Split**: Leave-One-Subject-Out (strict subject separation, no data leakage)  
**Train set**: 1,049 samples | **Test set**: 398 samples

### SPA-SNN Evolution (same dataset and split)

| Phase | Architecture | Val Accuracy | Learning Rule |
|---|---|---|---|
| Phase 1 | R-STDP baseline (single-scale RF, r=15) | 39.1% | R-STDP only |
| Phase 2 | Multi-scale RF (r=6,10,16) + SPA | 48.0% | STDP + SPA |
| Phase 3 | Hierarchical L1+L2 motif learning | 46.4% | STDP + SPA |
| **Final** | **Deep E-Prop (this repo)** | **89.2%** | **STDP + E-Prop** |

### CNN Baseline (different dataset — for reference only)

> ⚠️ The following was trained on **DailyDVS** (420 samples, no LOSO), not on DailyAction-DVS. Included for architectural comparison only.

| Model | Val Accuracy | Dataset | Learning Rule | Parameters |
|---|---|---|---|---|
| ResNet-18 | 60.7% | DailyDVS (420 samples) | Adam + BPTT | 11.6M |
| SNN (BPTT) | 46.0% | DailyDVS / PAFall | Backprop Through Time | ~2M |
| **SPA-SNN (ours)** | **89.2%** | **DailyAction-DVS LOSO** | **STDP + E-Prop** | **~2M** |

### Per-Class Performance (SOTA checkpoint)

| Class | Action | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 | Bend (Agacharse) | 1.00 | 0.86 | 0.93 |
| 1 | Climb (Escalar) | 0.94 | 1.00 | 0.97 |
| 2 | Falldown (Caerse) | 1.00 | 0.97 | 0.98 |
| 3 | Getup (Levantarse) | 0.95 | 1.00 | 0.97 |
| 4 | Jump (Saltar) | 0.94 | 1.00 | 0.97 |
| 5 | Liedown (Acostarse) | 0.90 | 0.97 | 0.93 |
| 6 | Carrybox (Cargar caja) | 0.93 | 0.83 | 0.88 |
| 7 | Run (Correr) | 0.90 | 0.90 | 0.90 |
| 8 | Sitdown (Sentarse) | 0.65 | 0.37 | 0.47 |
| 9 | Standup (Ponerse de pie) | 1.00 | 0.97 | 0.98 |
| 10 | Walk (Caminar) | 1.00 | 1.00 | 1.00 |
| 11 | Pickup (Recoger algo) | 0.55 | 0.80 | 0.65 |

**Overall accuracy: 89.2% | Macro F1: 0.89**

---

## Repository Structure

```
spa-snn/
├── Train_SPA_SNN.py          # Main training pipeline (STDP + E-Prop)
├── graph_snn.py              # GraphSNN architecture (LIF layers + local masks)
├── EProp_Deep.py             # Deep E-Prop classifier (BP-free, Bellec 2020)
├── STDP_improved.py          # STDP with biological noise suppression
├── Bichler_STDP.py           # Bichler 2012 STDP reference implementation
├── streaming_dataset.py      # Sliding-window DataLoader for HDF5 datasets
├── create_dailyaction_hdf5.py# DailyAction-DVS → HDF5 converter (with LOSO split)
├── saccade_augmenter.py      # Saccadic microsaccade augmentation
├── SPA_Classifier.py         # SPA (Segmented Probability-Maximization) classifier
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/villano99/spa-snn.git
cd spa-snn
pip install -r requirements.txt
```

---

## Dataset Preparation

Download **DailyAction-DVS** from the [official source](https://github.com/uzh-rpg/event-based_vision_resources)(https://drive.google.com/drive/folders/1JrYJnikaJdiNgq5Zz5pwbN-nwns-NNpz). Then convert to HDF5 with strict LOSO split:

```bash
python create_dailyaction_hdf5.py \
    --dataset-dir /path/to/DailyAction-DVS \
    --output dailyaction_ds_loso.h5
```

---

## Training

### Phase 1 — STDP Unsupervised Pre-training (L1)
```bash
python Train_SPA_SNN.py \
    --dataset dailydvs \
    --hdf5-path dailyaction_ds_loso.h5 \
    --epochs-stdp 5 \
    --save-w1 w1_stdp.pt \
    --downsample 1 \
    --l1-per-cell 2 \
    --l1-stride 2 \
    --num-classes 12
```

### Phase 2 — Deep E-Prop Supervised Training (L2/L3)
```bash
python Train_SPA_SNN.py \
    --dataset dailydvs \
    --hdf5-path dailyaction_ds_loso.h5 \
    --epochs-stdp 0 \
    --save-w1 w1_stdp.pt \
    --downsample 1 \
    --l1-per-cell 2 \
    --l1-stride 2 \
    --l2-size 4096 \
    --l3-size 1024 \
    --epochs-sup 300 \
    --batch-size 32 \
    --lr 0.1 \
    --num-classes 12 \
    --dense-head \
    --dropout 0.3 \
    --weight-decay 1e-4 \
    --ckpt-dir ckpts_sota
```

Best checkpoint is saved to `ckpts_sota/eprop_best.pt`.

---

## References

1. **E-Prop**: Bellec, G., et al. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications. https://doi.org/10.1038/s41467-020-17236-y

2. **STDP (Bichler 2012)**: Bichler, O., et al. (2012). *Extraction of temporally correlated features from dynamic vision sensors with spike-timing-dependent plasticity.* Neural Networks.

3. **DailyAction-DVS Dataset**: Liu, H., et al. (2021). *Event-Based Action Recognition Using Motion Information and Spiking Neural Networks.* IJCAI.

4. **SNNTorch**: Eshraghian, J. K., et al. (2023). *Training Spiking Neural Networks Using Lessons From Deep Learning.* Proceedings of the IEEE.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
