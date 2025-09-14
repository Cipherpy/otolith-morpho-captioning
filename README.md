# Otolith Morphology AI  

This repository provides tools and pipelines for **uploading, processing, and analyzing otolith images** to extract and explain morphological features. The project focuses on building **explainable AI systems** for fish taxonomy, with applications in species identification, ecological monitoring, and fisheries science.  

---

## 🚀 Features  

- **Image Upload Interface**  
  - Upload raw otolith images (`.jpg`, `.png`, `.tiff`).  
  - Automatic quality checks (size, resolution, grayscale conversion if needed).  

- **Morphological Feature Extraction**  
  - **Sulcus acusticus**  
  - **Cauda**  
  - **Ostium**  
  - **Posterior region**  
  - Additional margins (dorsal, ventral)  

- **AI-Powered Analysis**  
  - Captioning of features using **Gemma-3** and **LLaMA-3.2** finetuned models.  
  - Species-level prediction and classification.  
  - Out-of-distribution (OOD) stability checks.  

- **Visualization**  
  - Radial bar plots of feature frequency.  
  - Confusion matrices (row-normalized).    
---

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
