# Self-Supervised Temporal Representation Learning for Energy-Efficient sEMG Gesture Decoding

> **Project Type**  
> *Mini-Research Project · Proof-of-Thinking*  
>
> **Audience**  
> *Edge AI · Neuroengineering · Wearable Computing*  
>
> **Positioning**  
> *Mechanistic study of representation learning under system constraints — not benchmark chasing, not SoTA optimization.*

Wearable sEMG-based gesture recognition systems frequently underperform in real-world deployment—not because model capacity is insufficient, but because learned representations are brittle under **strong inter-subject variability** and **severely limited per-user labeling budgets**. In resource-constrained wearable systems, increasing model size is not a viable solution; instead, **representation quality under fixed compute and energy budgets** becomes the primary bottleneck.

This repository implements a **carefully controlled, leakage-aware experimental study** investigating whether **self-supervised temporal pretraining** can improve the quality of representations learned by **very small sEMG encoders**, such that downstream classification and user adaptation become more label-efficient **without increasing inference cost**.
