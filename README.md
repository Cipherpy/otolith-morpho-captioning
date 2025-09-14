# Otolith Morphology AI  

This repository provides tools and pipelines for **uploading, processing, and analyzing otolith images** to extract and explain morphological features. The project focuses on building **explainable AI systems** for fish taxonomy, with applications in species identification, ecological monitoring, and fisheries science.  

---

## 🚀 Features

1) **Out-of-Distribution (OOD) Detection**
   - Detects when an **input otolith belongs to a species not present in the training set**.
   - Helpful for **flagging candidate new records/species** or mislabeled samples.
   - Includes:
     - ID–OOD split utilities
     - Mahalanobis/kNN/energy-score baselines
     - Threshold tuning + reliability curves
     - Generalization gap metrics (𝐺 = M̄_ID − M̄_OOD)

2) **Morphology Feature Generation (VLM-based)**
   - **Upload an otolith image** and auto-generate **taxonomy-aligned morphological descriptors**:
     - *sulcus acusticus*, *ostium*, *cauda*, *posterior region*, margins.
   - Uses **finetuned Gemma-3** and **LLaMA-3.2** captioners to produce concise, expert-style text.
   - Outputs:
     - Structured JSON of features (for downstream analysis)
     - Human-readable captions (for reports/manuscripts)
     - Optional overlays/figures for presentations
---

##**Sample caption output**

```
caption: "Type: Sagittal. Side: Right otolith. Shape: oval, sinuate to creante dorsal and serrate entral margins. Sulcus acusticus: heterosulcoid, ostial, median. Ostium: Funnel-like. Cauda: tubular, strongly curved. Anterior region: blunt, rostrum blunt and antirostrum blunt or poorly developed, excisura wide with wide, shallow notch. Posterior region: blunt."


## 📂 Repository Structure  


---
- **cnn/** → for CNN baselines (ResNet, VGG, etc.).  
- **captioning/** → for VLM-based morphological description.  
- **ood/** → for OOD splits and generalization gap calculations.  



## 📖 Citation  

If you use this repository in your research, please cite:  


## Contact
For questions or issues, please open an issue in this repository or contact the maintainers

#### Name: Reshma B
#### Email: reshmababuraj89@gmail.com
