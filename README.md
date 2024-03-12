# FastGlioma: visual foundation models for fast, label-free detection of diffuse glioma infiltration

[**Preprint**](https://www.researchsquare.com/article/rs-4033133/v1) /
[**Interactive Demo**](https://fastglioma.mlins.org) /
**Models** (coming soon!) /
[**MLiNS Lab**](https://mlins.org)
 
Code repository for our paper 'Visual foundation models for fast, label-free detection of diffuse glioma infiltration.' We employ a foundational model training strategy to predict the degree of diffuse glioma infiltration intraoperatively using stimulated Raman histology and deep learning.

## Abstract

A critical challenge in diffuse glioma treatment is detecting tumor infiltration during surgery to achieve safe maximal resection. Unfortunately, safely resectable residual tumor is found in the majority of glioma patients after surgery, leading to early recurrence and decreased patient survival. We present **FastGlioma**, a visual foundation model for fast (<10 seconds) and accurate detection of glioma infiltration in fresh, unprocessed surgical tissue. FastGlioma was pretrained using large-scale self-supervision (~4 million images) on rapid, label-free, optical microscopy, and fine-tuned to output a normalized score that indicates the degree of tumor infiltration within whole slide optical images. In a prospective, multicenter, international testing cohort of diffuse glioma patients (n=220), FastGlioma was able to detect and quantify the degree of tumor infiltration with an average area under the ROC curve of 92.1 &pm; 0.9\%. FastGlioma outperformed image-guided and fluorescence-guided adjuncts for detecting tumor infiltration during surgery by a wide margin in a head-to-head, prospective comparison study. FastGlioma performance remained high across diverse patient demographics, medical centers, and diffuse glioma subtypes as defined by the World Health Organization (WHO). FastGlioma shows zero-shot generalization to other adult and pediatric brain tumor diagnoses, demonstrating that our visual foundation model can serve as a general-purpose adjunct for guiding brain tumor surgeries. These findings represent the transformative potential of medical foundation models to unlock the role of artificial intelligence in the care of cancer patients.

## TL;DR

*Image tumor with **Fast SRH** >> **FastGlioma** >> Degree of tumor infiltration out* (end-to-end: ~10 seconds)

## Installation

1. Clone FastGlioma github repo
   ```console
   git clone git@github.com:MLNeurosurg/fastglioma.git
   ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment
    ```console
    conda create -n fastglioma python=3.9
    ```
4. Activate conda environment
    ```console
    conda activate fastglioma
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/fastglioma/repo/dir>
    pip install -e .
    ```

## Directory organization
```
fastglioma/
├── fastglioma/         # library for FastGlioma training
│   ├── datasets/       # PyTorch datasets
│   ├── losses/         # FastGlioma loss functions with contrastive/ordinal metric learning
│   ├── models/         # PyTorch models for training and evaluation
│   └── train/          # Training and evaluation scripts
│       └── config/     # Configuration files used for training
├── figures/            # Figures in the README file
├── README.md
├── setup.py            # Setup file including list of dependencies
├── LICENSE             # MIT license for the repo
└── THIRD_PARTY         # License information for third party code
```

# Overview

![Overview](/figures/Figure_1.png)

**FastGlioma workflow.** A patient with a suspected diffuse glioma undergoes surgical resection. During tumor resection, the surgeon samples
tissue from the surgical margin. The portable SRH imaging system acquires microscopic images in the operating room, performed by a single technician
using simple touchscreen instructions. A freshly excised surgical specimen is loaded directly into a custom microscope slide and inserted into the SRH
imager without the need for tissue processing. SRH images can be virtually
stained using an H&E-like colorscheme for clinician review as shown above. A whole slide SRH image is divided into patches and each patch undergoes
a feedforward pass through a patch tokenizer. The patch tokens, plus an appended classification token <CLS>, are then input into
a whole slide SRH encoder that is a vision transformer. The patch tokenizer and whole slide encoder are pretrained as a visual foundation model using
large-scale self-supervision. For tumor infiltration scoring, a slide scorer layer is fine-tuned to output a normalized score between
0-1 that predicts the degree of tumor infiltration within the whole slide image. Additionally, FastGlioma provides real-time
regional or field-of-view interpretability by identifying areas of high tumor infiltration within whole slide images.

# Results

![Results](/figures/Figure_2.png)

**FastGlioma performance.** a, Prediction results for the full prospective, international, multicenter testing cohort of diffuse gliomas
patients (n = 220) are shown. ROC curves show average performance for predicting four levels of tumor infiltration. SRH foundation model pretraining showed strong prediction performance without fine-tuning. FastGlioma that included fine-tuning with ordinal metric learning had a 3.2% increase in overall performance. b, Boxplots of FastGlioma infiltration scores by ground truth value are shown. Scores had strong correlation with ground truth ordinal scores (ρ = 0.77 95% confidence interval 0.74-0.78). c, FastGlioma performance on full resolution versus
low resolution SRH images is shown. FastGlioma allows for 10X increase in imaging speed with minimal performance tradeoff. d, Whole slide SRH
representations are plotted on a linear discriminant axis. FastGlioma learned representations that rank whole slide SRH images on a near-linear tumor
infiltration axis. e, Subgroup analysis by WHO adult-type diffuse glioma subtypes. FastGlioma performs well across all three adult-type diffuse gliomas.
Importantly, FastGlioma performs well on lower grade gliomas where tumor infiltration and tissue cellularity can be low. Lower
grade and lower tumor infiltration are major challenges for other surgical adjuncts, such as fluorescence-guided surgery.

© This code is made available for academic purposes. Imaging and clinical information for this project was collected with IRB approval (HUM00083059) and is protected under HIPAA. Representative images and predictions can be found at [**fastglioma.mlins.org**](https://fastglioma.mlins.org).

## License Information
The code is licensed under the MIT License.
See LICENSE for license information and third party notices.