# 🐟 Otolith-Morpho-Captioning
### Morphology-Grounded Vision–Language Modeling for Taxonomic Explainability

## 🔍 Problem

Taxonomic identification in ichthyology is increasingly dependent on image-based pipelines. However, existing models often fail to deliver morphology-grounded explainability and suffer severe performance degradation under domain shifts (e.g., changes in camera setup, illumination, geography, or out-of-distribution species).

### 🎯 Goal

This repository provides research-grade code and reproducible protocols for training and evaluating vision–language models (VLMs) that generate diagnostic, region-referential captions for otolith images. The system is designed to:

- Ground textual outputs in morphological regions (sulcus, ostium, cauda, posterior margin).

- Couple captioning with species identification tasks.

- Benchmark robustness under open-world and OOD (out-of-distribution) settings.

- Built for researchers aiming to push the frontier of explainable multimodal AI in taxonomy, ecology, and fisheries science.

---

## 🚀 Features  

### 1. Out-of-Distribution (OOD) Detection  
- Identifies when an **otolith belongs to a species not present in the training dataset**.  
- Useful for **flagging potential new species** or mislabeled records.  
- Includes:  
  - ID–OOD split utilities  
  - Mahalanobis / kNN / energy-score baselines  
  - Threshold tuning + reliability curves  
  - Generalization gap metrics (𝐺 = M̄_ID − M̄_OOD)  

### 2. Morphology Feature Generation (VLM-based)  
- **Upload an otolith image** and automatically generate **taxonomy-aligned morphological descriptors**:  
  - *sulcus acusticus*, *ostium*, *cauda*, *posterior region*, margins.  
- Uses **finetuned Gemma-3** and **LLaMA-3.2** captioners to produce concise, expert-style text.  
- Outputs:  
  - Structured JSON of features (for downstream analysis)  
  - Human-readable captions (for reports/manuscripts)  
  - Optional overlays/figures for presentations  

---

## 📝 Sample Caption Output  

---
caption: "Type: Sagittal. Side: Right otolith. Shape: oval, sinuate to crenate dorsal and serrate ventral margins. 
Sulcus acusticus: heterosulcoid, ostial, median. 
Ostium: funnel-like. 
Cauda: tubular, strongly curved. 
Anterior region: blunt, rostrum blunt and antirostrum blunt or poorly developed, excisura wide with wide, shallow notch. 
Posterior region: blunt."

----

## 📂 Repository Structure  
otolith-morpho-captioning/
│
├── cnn/          # CNN baselines (ResNet, VGG, etc.)
├── captioning/   # VLM-based morphological description (Gemma, LLaMA)
├── ood/          # OOD splits, detection, and generalization gap analysis
├── scripts/      # Preprocessing and shared utilities
├── notebooks/    # Interactive Jupyter notebooks
├── models/       # Trained checkpoints
├── outputs/      # Visualizations and results
└── data/         # Example otolith images and annotations

## 📖 Citation
---
If you use this repository in your research, please cite:
@article{Reshma2025OtolithAI,
  title   = {Automated Otolith Morphology Analysis using AI},
  author  = {Reshma B and Collaborators},
  journal = {Under Submission},
  year    = {2025}
}

## 📬 Contact

For questions, suggestions, or issues, please open an Issue
 in this repository or contact the maintainer:

Name: Reshma B
Email: reshmababuraj89@gmail.com



