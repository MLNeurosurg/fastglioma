# FastGlioma: foundation models for fast, label-free detection of glioma infiltration

[**Preprint**](https://www.researchsquare.com/article/rs-4033133/v1) /
[**Interactive Demo**](https://fastglioma.mlins.org) /
[**Models**](https://huggingface.co/mlinslab/fastglioma) /
[**MLiNS Lab**](https://mlins.org)
 
Code repository for our paper 'Foundation models for fast, label-free detection of glioma infiltration.' We employ a foundational model training strategy to predict the degree of diffuse glioma infiltration intraoperatively using stimulated Raman histology and deep learning.

## TL;DR

*Image tumor with **Fast SRH** >> **FastGlioma** >> Degree of tumor infiltration out* (end-to-end: ~10 seconds)

## Abstract

A critical challenge in glioma treatment is detecting tumor infiltration during surgery to achieve safe maximal resection. Unfortunately, safely resectable residual tumor is found in the majority of glioma patients after surgery, causing early recurrence and decreased survival. We present **FastGlioma**, a visual foundation model for fast (<10 seconds) and accurate detection of glioma infiltration in fresh, unprocessed surgical tissue. FastGlioma was pretrained using large-scale self-supervision (∼4 million images) on rapid, label-free, optical microscopy, and fine-tuned to output a normalized score that indicates the degree of tumor infiltration within whole slide optical images. In a prospective, multicenter, international testing cohort of diffuse glioma patients (n=220), FastGlioma was able to detect and quantify the degree of tumor infiltration with an average area under the ROC curve of 92.1 &pm; 0.9\%. FastGlioma outperformed image-guided and fluorescence-guided adjuncts for detecting tumor infiltration during surgery by a wide margin in a head-to-head, prospective study (n=129). FastGlioma performance remained high across diverse patient demographics, medical centers, and diffuse glioma molecular subtypes as defined by the World Health Organization (WHO). FastGlioma shows zero-shot generalization to other adult and pediatric brain tumor diagnoses, demonstrating the potential for our foundation model to serve as a general-purpose adjunct for guiding brain tumor surgeries. These findings represent the transformative potential of medical foundation models to unlock the role of artificial intelligence in the care of cancer patients.

## Intended Use
*FastGlioma is for investigational use only*. FastGlioma is intended for patients who have adult-type diffuse gliomas as defined by the World Health Organization (WHO). These include:

  1. Astrocytoma, IDH-mutant
  
  2. Oligodendroglioma, IDH-mutant, and 1p/19q-codeleted
  
  3. Glioblastoma, IDH-wildtype
  
Study neurosurgeons were allowed to include patients based on (1) a previous pathologic diagnosis of adult-type diffuse glioma or (2) high likelihood of adult-type diffuse glioma diagnosis based on clinical presentation and radiographic features. Intraoperative pathologic diagnosis via frozen sectioning or SRH imaging was completed in the majority of patients to provide further preliminary evidence of diffuse glioma diagnosis prior to margin sampling for FastGlioma. While our preliminary data show good zero-shot performance on a variety of other tumors and clinical settings, FastGlioma is **not** intended for surgical resection guidance around eloquent cortical or subcortical structures, pediatric patients, non-primary brain tumors, or non-neoplastic pathologic tissue.

FastGlioma was trained using ordinal labels that correspond four increasing degrees of tumor infiltration: 0, 1, 2, or 3. However, because tumor infiltration is a continuous variable, FastGlioma outputs a continuous normalized score between 0-1 to indicate the degree of tumor infiltration. Based on training and testing results, we recommend guidelines regarding FastGlioma scores:

| Pathologists Score | FastGlioma range | Interpretation |
|----------|----------|----------|
| Score 0 | 0-25% | Normal or non-neoplastic tissue |
| Score 1 | 26-50% | Atypical cells, cannot rule out tumor |
| Score 2 | 51-85% | Sparse tumor infiltration |
| Score 3 | 86-100% | Dense tumor infiltration |

Please note that the nontumor-tumor threshold corresponds to a FastGlioma score of 50%. We hope to provide surgeon's with real-time, accurate, and clinically actionable diagnostic information. Ultimately, the decision to resect additional tissue we leave to the operating surgeon and the clinical context.

## Overview

![Overview](/figures/Figure_1.png)

**FastGlioma workflow.** A patient with a suspected diffuse glioma undergoes surgical resection. During tumor resection, the surgeon samples tissue from the surgical margin. The portable SRH imaging system acquires microscopic images in the operating room, performed by a single technician using simple touchscreen instructions. A freshly excised surgical specimen is loaded directly into a custom microscope slide and inserted into the SRH imager without the need for tissue processing. Additional details on image acquisition can be found in Extended Data Fig. 1. SRH images can be virtually stained using an H&E-like colorscheme for clinician review as shown above. A whole slide SRH image is divided into patches and each patch undergoes a feedforward pass through a patch tokenizer (Extended Data Fig. 3a). The patch tokens, plus an appended classification token <CLS>, are then input into a whole slide SRH encoder that is a vision transformer. The patch tokenizer and whole slide encoder are pretrained as a visual foundation model using large-scale self-supervision (Extended Data Fig. 3b). For tumor infiltration scoring, a slide scorer model is fine-tuned to output a normalized continuous score between 0-1 that predicts the degree of tumor infiltration within the whole slide image that corresponds to a 4-tier whole slide ordinal infiltration scale as defined by expert neuropathologists (Extended Data Fig. 2 and 4). Ordinal labels are weak because they apply to the slide-level only. Despite the weak labels, FastGlioma provides regional interpretability by identifying area within whole slides SRH images with high probability of tumor infiltration. Scale bars, 100 microns.

## Results

![Results](/figures/Figure_2.png)

**FastGlioma performance.** a, Prediction results for the full prospective, international, multicenter testing cohort of diffuse gliomas patients (n = 220) are shown. ROC curves (plotted as mean &pm; s.d.) show average performance for predicting four levels of tumor infiltration. See Extended Data Fig 6 for subgroup analysis. SRH foundation model pretraining showed strong prediction performance without fine-tuning. FastGlioma that included fine-tuning with ordinal metric learning had a 3.2\% increase in overall performance. FastGlioma outperforms models trained using standard supervised training (84.7 &pm; 1.1\% mAUC) as shown in Supplementary Data Table 4. b, Box and whisker plots, shown in the standardized quartile format, of FastGlioma infiltration scores by ground truth value are shown. Scores had strong correlation with ground truth ordinal scores (ρ = 0.77 95\% confidence interval 0.74-0.78). Individual scores are shown in a histogram and correspond to AUROC values in 2a. c, FastGlioma performance on full resolution versus low resolution SRH images is shown (plotted as mean &pm; s.d.). FastGlioma allows for 10X increase in imaging speed with minimal performance tradeoff. d, Whole slide SRH representations are plotted on a linear discriminant axis. FastGlioma learned representations that rank whole slide SRH images on a near-linear tumor infiltration axis. e, Subgroup analysis by WHO adult-type diffuse glioma subtypes (ROC curves plotted as mean &pm; s.d.). FastGlioma performs well across all three adult-type diffuse gliomas. Importantly, FastGlioma performs well on lower grade gliomas where tumor infiltration and tissue cellularity can be low (Extended Data Fig. 7). Low grade and lower tumor infiltration are major challenges for other surgical adjuncts, such as fluorescence-guided surgery.

© This code is made available for academic purposes. Imaging and clinical information for this project was collected with IRB approval (HUM00083059) and is protected under HIPAA. Representative images and predictions can be found at [**fastglioma.mlins.org**](https://fastglioma.mlins.org).

# Training, evaluation, and inference

This repository currently supports inference on the [OpenSRH dataset](https://opensrh.mlins.org/), the largest publically available stimulated Raman histology dataset, with FastGlioma models available on [HuggingFace](https://huggingface.co/mlinslab/fastglioma/). Training/evaluation scripts will be released soon.

## Directory organization
```
fastglioma/
├── fastglioma/         # Library for FastGlioma training
│   ├── datasets/       # PyTorch OpenSRH datasets
│   ├── losses/         # FastGlioma loss functions with contrastive/ordinal metric learning
│   ├── models/         # PyTorch models for training, evaluation, and inference
│   ├── utils/          # Utility functions
│   ├── inference/      # Inference scripts
│       └── config/     # Configuration files used for inference
├── figures/            # Figures in the README file
├── THIRD_PARTY         # License information for third party code   
├── setup.py            # Setup file including list of dependencies
├── LICENSE             # MIT license for the repo
└── README.md           
```

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

## Dataset and Models

The OpenSRH dataset and FastGlioma models are available for non-commerical use. Please download the OpenSRH dataset from the [OpenSRH website](https://opensrh.mlins.org/) according to the instructions provided. Additionally, please request access to FastGlioma models on [Hugging Face](https://huggingface.co/mlinslab/fastglioma/).

## Inference

1. Log into Hugging Face
    ```console
    huggingface-cli login
    ```
2. Specify inference configuration file
    ```console
    vi fastglioma/inference/config/infer.yaml
    ```
3. Generate predictions
    ```console
    python fastglioma/inference/run_inference.py -c fastglioma/inference/config/infer.yaml
    ```

## License Information
The code is licensed under the MIT License.
See LICENSE for license information and THIRD_PARTY for third party notices.