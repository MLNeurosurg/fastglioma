# FastGlioma: visual foundation models for fast, label-free detection of diffuse glioma infiltration

[**Preprint**](https://arxiv.org) /
[**Interactive Demo**](https://fastglioma.mlins.org) /
[**Models**](https://huggingface.co) /
[**MLiNS Lab**](https://mlins.org)
 
Code repository for our paper 'Visual foundation models for fast, label-free detection of diffuse glioma infiltration.' We employ a foundational model training strategy to predict the degree of diffuse glioma infiltration intraoperatively using stimulated Raman histology and deep learning.

## Abstract

A critical challenge in diffuse glioma treatment is detecting tumor infiltration during surgery to achieve safe maximal resection. Unfortunately, safely resectable residual tumor is found in the majority of glioma patients after surgery, leading to early recurrence and decreased patient survival. We present \textit{\textbf{FastGlioma}}, a visual foundation model for fast (<10 seconds) and accurate detection of glioma infiltration in fresh, unprocessed surgical tissue. FastGlioma was pretrained using large-scale self-supervision (~4 million images) on rapid, label-free, optical microscopy, and fine-tuned to output a normalized score that indicates the degree of tumor infiltration within whole slide optical images. In a prospective, multicenter, international testing cohort of diffuse glioma patients (n=220), FastGlioma was able to detect and quantify the degree of tumor infiltration with an average area under the ROC curve of 92.1 &pm; 0.9\%. FastGlioma outperformed image-guided and fluorescence-guided adjuncts for detecting tumor infiltration during surgery by a wide margin in a head-to-head, prospective comparison study. FastGlioma performance remained high across diverse patient demographics, medical centers, and diffuse glioma subtypes as defined by the World Health Organization (WHO). FastGlioma shows zero-shot generalization to other adult and pediatric brain tumor diagnoses, demonstrating that our visual foundation model can serve as a general-purpose adjunct for guiding brain tumor surgeries. These findings represent the transformative potential of medical foundation models to unlock the role of artificial intelligence in the care of cancer patients.

## TL;DR

*Image tumor with **SRH** >> Images in >> **FastGlioma** >> Degree of tumor infiltration out* (end-to-end: ~10 seconds)

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

© This code is made available for academic purposes. Imaging and clinical information for this project was collected with IRB approval (HUM00083059) and is protected under HIPAA. Representative images and predictions can be found at [**fastglioma.mlins.org**](https://fastglioma.mlins.org).

## License Information
The code is licensed under the MIT License.
See LICENSE for license information and third party notices.