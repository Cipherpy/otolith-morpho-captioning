Otolith Morphology AI

This repository provides tools and pipelines for uploading, processing, and analyzing otolith images to extract and explain morphological features. The project focuses on building explainable AI systems for fish taxonomy, with applications in species identification, ecological monitoring, and fisheries science.

🚀 Features

Image Upload Interface

Upload raw otolith images (.jpg, .png, .tiff).

Automatic quality checks (size, resolution, grayscale conversion if needed).

Morphological Feature Extraction

Sulcus acusticus

Cauda

Ostium

Posterior region

Additional margins (dorsal, ventral)

AI-Powered Analysis

Captioning of features using Gemma-3 and LLaMA-3.2 finetuned models.

Species-level prediction and classification.

Out-of-distribution (OOD) stability checks.

Visualization

Radial bar plots of feature frequency.

Confusion matrices (row-normalized).

Nature-style figures with high-resolution export.

📂 Repository Structure
otolith-morpho-captioning/
│
├── data/                 # Example otolith images & annotations
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Image preprocessing & feature extraction
├── models/               # Trained models (Gemma-3, LLaMA-3.2 adapters)
├── outputs/              # Visualizations & results
└── README.md             # Project overview

🖼️ Uploading Otolith Images

Place your otolith images inside the data/ folder.

Supported formats: JPEG, PNG, TIFF.

Run the preprocessing script:

python scripts/preprocess_images.py --input data/ --output processed/


This will:

Normalize image sizes.

Apply contrast adjustment.

Segment otolith regions of interest.

🔬 Feature Explanation

Each otolith image is analyzed for the following diagnostic regions:

Sulcus acusticus → central groove critical for species differentiation.

Cauda → posterior narrowing of the sulcus, shape and length vary by species.

Ostium → anterior widening of the sulcus, useful for identifying genera.

Posterior region → overall end shape of the otolith, important for classification.

AI models generate captioned descriptions of these features to align with taxonomic keys.

📊 Example Outputs

Radial plots: frequency of feature terms across species.

Confusion matrices: model prediction quality vs. actual species.

Caption samples:

“The sulcus acusticus is moderately curved, ostium rounded, cauda elongated — consistent with Diaphus arabicus.”

⚙️ Requirements

Python ≥ 3.9

PyTorch ≥ 2.0

Transformers (Hugging Face)

OpenCV

Matplotlib / Seaborn / Cartopy

Install dependencies:

pip install -r requirements.txt

📖 Citation

If you use this repository in your research, please cite:

@article{YourName2025OtolithAI,
  title   = {Automated Otolith Morphology Analysis using AI},
  author  = {Your Name and Collaborators},
  journal = {Under Submission},
  year    = {2025}
}

🤝 Contributing

Contributions are welcome! Please open an issue or pull request for:

New otolith datasets

Additional feature extraction modules

Improvements in visualization
