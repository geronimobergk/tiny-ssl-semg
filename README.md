# Self-Supervised Temporal Representation Learning for Energy-Efficient sEMG Gesture Decoding

**Status:** Ongoing research project **(WIP)**

**Type:** Mini research project, Proof-of-thinking

**Focus:** Edge AI, Neuroengineering, Wearable computing

**Positioning:** Mechanistic study of representation learning under deployment constraints

## Overview

Wearable sEMG-based gesture recognition systems frequently underperform in real-world deployment—not because model capacity is insufficient, but because learned representations are brittle under **strong inter-subject variability** and **severely limited per-user labeling budgets**. In resource-constrained wearable systems, increasing model size is not a viable solution; instead, **representation quality under fixed compute and energy budgets** becomes the primary bottleneck.

This project investigates whether **self-supervised pretraining (SSL)** can improve the quality of temporal representations learned by **very small sEMG encoders**, such that:

* downstream classification becomes easier,
* user adaptation becomes more label-efficient, and
* inference cost remains unchanged.

The project is deliberately scoped to **isolate causal effects**, rather than to maximize benchmark performance.

## Central Hypothesis

> For a fixed tiny encoder architecture and a fixed inference budget, self-supervised pretraining improves
> **(a)** linear separability of sEMG representations and
> **(b)** label-efficient user adaptation,
> **without increasing inference cost**.

### Operationalization (Success Criteria)

* **H1 — Representation Quality**
  
  A *frozen* SSL-pretrained encoder with a linear probe achieves higher cross-subject performance than a supervised encoder trained from scratch, evaluated under **identical linear probing conditions**.

* **H2 — Adaptation Efficiency**
  
  SSL-pretrained representations require less calibration data (5 – 30 s) to reach a fixed target performance compared to supervised representations.

* **H3 — System Validity**

  Encoder parameters, MACs/sec, and batch-1 latency remain identical across supervised and SSL variants.

Negative or neutral outcomes are **explicitly acceptable and informative**.

## Scope

The study is deliberately constrained to a single dataset, a fixed tiny encoder architecture, and one controlled self-supervised intervention in order to enable clear causal attribution under realistic wearable-system constraints.

### Dataset

* NinaPro DB2
* sEMG only (12 channels, 2 kHz)

### Gesture Set

* Exercise B (17 gestures)
* Widely used, manageable, and representative

### Signal Processing

* Window length: 200 ms (400 samples)
* Hop size: 10 ms
* Raw signal only (no handcrafted features, no spectral transforms -> minimal processing)

### Explicit Exclusions

* DB3, multimodal fusion, or cross-dataset transfer
* Transformers, attention mechanisms, large encoders
* Continual learning or online learning

## Evaluation Protocols

To ensure that performance differences can be causally attributed to representation learning rather than data leakage, this project employs evaluation protocols that are explicitly designed to be leakage-proof. Splits, normalization, and adaptation procedures are defined at the subject and time-block level, mirroring realistic deployment conditions and preventing optimistic bias.

### A. Cross-Subject Generalization (Primary Protocol)

This protocol evaluates generalization to a completely unseen user, with zero labeled or unlabeled exposure to the test subject during training.

* Train: N−1 subjects
* Test: 1 held-out subject (LOSO)
* No labeled *or unlabeled* data from the test subject during training

### B. User Adaptation

This protocol evaluates post-deployment personalization under a strictly limited labeling budget, where adaptation is performed on a small, fixed-duration calibration segment and performance is measured on the remaining data from the same user.

* Calibration data: 5 s / 10 s / 30 s
* Adaptation performed **only** on calibration data
* Evaluation on remaining data from the same subject

### Hard Leakage Safeguards

1. **Split-local normalization**

   * Cross-subject: statistics from training subjects only
   * Adaptation: statistics from calibration data only

2. **Time-block-based splits**

   * Calibration and evaluation data are contiguous time blocks
   * No window-level randomization
   * Transition windows at gesture boundaries discarded

3. **SSL pretraining variants**

   * Primary: pretraining on training subjects only
   * Secondary (upper bound): pretraining on full unlabeled dataset
     *(clearly labeled and separated in reporting)*

## Model Architecture

The model architecture is intentionally fixed, minimal, and fully auditable in order to isolate the effect of representation learning and prevent confounding capacity-driven improvements.

### Encoder

The encoder serves as the sole study object and is designed to be compact and deployment-relevant.

* Tiny Temporal Convolutional Network (TCN)
* 3–4 convolutional blocks
* Depthwise-separable 1D convolutions
* GroupNorm (batch-size invariant)
* ReLU or GELU activations
* Global average pooling

### Hard Constraints

* < 50 k parameters
* Receptive field ≥ 200 ms
* MACs/sec explicitly computed from window and hop size

### Classifier

* Linear softmax head
* No MLPs, no attention

**Rationale:**
The architecture is intentionally *boring* to prevent confounding effects and to keep the study focused on **representation learning**, not model expressivity.

## Self-Supervised Pretraining

Self-supervised learning is introduced as a single, controlled intervention to improve representation quality without changing the encoder architecture or inference cost.

**Method:** Masked Signal Modeling (MSM)

### Pretext Task

During pretraining, a fixed fraction of the input signal is masked and the model is trained to reconstruct the missing samples.

* Mask 10–20 % contiguous time segments
* Optional full-channel masking (single ablation only)
* Encoder → latent representation
* Lightweight decoder reconstructs masked samples
* Loss: Mean Squared Error (MSE)

### Design Rationale

Masked Signal Modeling is selected for its simplicity, stability, and suitability for small-data regimes typical of wearable sEMG. 
As a label-free objective, it enables pretraining without additional annotation effort while encouraging the encoder to capture temporal continuity and inter-channel structure directly from the raw signal.
The use of a lightweight decoder during pretraining ensures that reconstruction capacity does not leak into downstream inference, as the decoder is discarded after pretraining and does not affect deployment cost.
A single, well-defined masking strategy is used to keep the intervention minimal, interpretable, and causally attributable.

## Comparison Conditions (Causally Controlled)

To isolate the effect of the pretraining signal, all comparison conditions use the same encoder architecture, classifier head, optimization setup, evaluation protocol, and inference budget.
The only variable across conditions is how the encoder is trained prior to evaluation.

### Baseline A — Supervised Training from Scratch

The encoder and linear classifier are trained **end-to-end** using labeled data only.

* Identical tiny encoder and linear softmax head
* Supervised optimization with cross-entropy loss

This baseline represents the standard fully supervised learning setup.

### Baseline B — Compute-Matched Supervised Training

This baseline controls for the possibility that observed SSL gains are driven by **greater optimization exposure**, rather than improved representation learning.

The supervised model is trained end-to-end using the **same encoder and linear head**, but the **total number of encoder parameter updates is matched exactly** to those performed in the SSL pipeline (i.e., SSL pretraining updates plus subsequent linear-probe training).

* Identical encoder architecture and classifier head
* Encoder update budget matched to SSL (pretraining + probing)
* Same optimization setup and inference budget

By holding the **encoder optimization budget constant**, this condition isolates the effect of the **pretraining signal itself**. Any remaining performance difference can therefore be attributed to changes in representation quality, not to longer training or increased optimization effort.

### SSL Variant — Self-Supervised Pretraining + Linear Probe

The encoder is first trained using a self-supervised objective, then evaluated under a frozen-encoder protocol.

* Self-supervised pretraining via masked signal modeling (MSM)
* Decoder discarded after pretraining
* Encoder weights frozen
* Linear classifier trained on labeled data only

This setup directly tests whether SSL improves the **quality and usability of learned representations** under a fixed inference budget.

### Representation Fairness Check (Primary for H1)

To ensure a fair comparison of representations, supervised and SSL encoders are evaluated under **identical probing conditions**.

* Train a supervised model end-to-end
* Freeze the trained encoder
* Train a new linear probe using the **same protocol** as for the SSL encoder

This comparison ensures that differences in performance reflect **representation quality**, not differences in classifier capacity or training procedure.

## Metrics & Analyses

### Performance Metrics

Model performance is evaluated using **class-balanced metrics** to account for gesture imbalance and inter-subject variability.

* **Macro-F1** is used as the **primary decision metric**, as it weights all gesture classes equally and reflects per-class discriminability.
* **Balanced Accuracy** is reported as a complementary metric to ensure consistency and interpretability across evaluation protocols.

All performance metrics are computed under identical evaluation settings across comparison conditions.

### System Metrics

To verify that representational gains do not come at increased system cost, each model is evaluated using explicit **efficiency metrics**.

* **Parameter count** to ensure architectural equivalence
* **MACs per second**, derived from window length and hop size, to quantify computational load at inference time
* **Batch-1 CPU latency** measured under fixed hardware and runtime conditions to reflect on-device performance

### Representation Analysis

To directly examine changes in learned representations, embedding-level analyses are performed.

* **Intra- vs. inter-subject feature variance** to assess subject invariance and separability
* Comparison of representations **before and after user adaptation** to analyze adaptation dynamics
* **Uniform Manifold Approximation and Projection (UMAP) visualizations** are included for qualitative illustration

### Key Figures (Exactly Four)

1. **Cross-subject performance:** supervised vs. SSL representations evaluated via frozen encoders and linear probes
2. **Accuracy vs. calibration time:** performance as a function of available user calibration data (5 / 10 / 30 s)
3. **Accuracy vs. MACs/sec:** performance plotted against computational cost to highlight efficiency trade-offs
4. **Representation variance:** changes in intra- and inter-subject feature variance before and after adaptation

## Deployment Awareness

To consider deployment constrains, the following is implemented in additon:

* Export encoder to ONNX
* Apply INT8 post-training quantization
* Measure batch-1 latency on a fixed CPU target
* Report relative latency and accuracy deltas

## Expected Outcomes

* SSL improves cross-subject linear separability
* SSL reduces required calibration data
* Inference cost remains unchanged
