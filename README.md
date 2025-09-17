# Experiments Summary for Robust Indian Language Identification

This document provides a detailed overview of the experiments conducted in the study on robust spoken language identification (LID) across 12 Indian languages. It outlines the objectives, configurations, datasets, features, models, and key outcomes of each experimental setup.

---

## Table of Contents
- [Introduction](#introduction)
- [Experiment 1: Baseline Systems on In-Domain and Out-of-Domain Data](#experiment-1-baseline-systems-on-in-domain-and-out-of-domain-data)
- [Experiment 2: Fine-tuning of Off-the-shelf Pretrained Models](#experiment-2-fine-tuning-of-off-the-shelf-pretrained-models)
- [Experiment 3: Proposal of Phonotactic Feature-based Models](#experiment-3-proposal-of-phonotactic-feature-based-models)
- [Experiment 4: Comparison of Models](#experiment-4-comparison-of-models)
- [Experiment 5: Visualizing Language Clusters (t-SNE Analysis)](#experiment-5-visualizing-language-clusters-t-sne-analysis)
- [Summary of Results](#summary-of-results)
- [References](#references)

---

## Introduction

The experiments aim to evaluate the effectiveness of various features and models for language identification in Indian languages, under different domain conditions, resource constraints, and model complexities. The core focus is on assessing traditional acoustic features versus phonotactic features derived from self-supervised models, with an emphasis on robustness to domain shifts.

---

## Experiment 1: Baseline Systems on In-Domain and Out-of-Domain Data

**Objective**  
Compare traditional and deep learning-based baseline systems on both seen (in-domain) and unseen (out-of-domain) test data.

**Models**
- **Bxvec**: X-vector system with TDNN architecture, trained on 80-dimensional bottleneck features.  
- **BmfccConf**: Conformer model trained on MFCCs (39-dimensional features, including derivatives).  
- **Bw2v2Conf**: Conformer model trained on wav2vec 2.0 embeddings.  

**Data**
- **In-Domain (Seen):** Test set from training domains.  
- **Out-of-Domain (Unseen):** Test set from unseen datasets (different environments, spontaneous speech).  

**Results**

| Model       | Seen Accuracy (%) | Seen EER (%) | Unseen Accuracy (%) | Unseen EER (%) |
|-------------|------------------:|-------------:|--------------------:|---------------:|
| Bxvec       | 80.6              | 8.6          | 51.2                | 19.5           |
| BmfccConf   | 92.7              | 3.4          | 17.7                | 40.6           |
| Bw2v2Conf   | 92.4              | 3.9          | 29.9                | 33.1           |

**Key Findings**
- MFCC-based conformers perform best on in-domain data.  
- All models degrade on unseen data; MFCCs show the sharpest drop.  
- Deep acoustic models are less robust to domain shifts.  

---

## Experiment 2: Fine-tuning Off-the-shelf Pretrained Models

**Objective**  
Evaluate how large pre-trained models like OpenAI’s Whisper-base and VoxLingua107 ECAPA-TDNN adapt to Indian language data through fine-tuning.

**Setup**
- Whisper: last four encoder layers fine-tuned on train data.  
- VoxLingua107 ECAPA-TDNN: trained directly on the Indian dataset.  

**Results**

| Model                        | Seen Accuracy (%) | Unseen Accuracy (%) | Comments                                  |
|-------------------------------|------------------:|--------------------:|-------------------------------------------|
| Whisper-base (fine-tuned)     | 97.9              | 88.1                | High accuracy, resource-heavy             |
| VoxLingua107 ECAPA-TDNN       | 44.4              | 15.4                | Poor generalization, domain mismatch      |

**Key Findings**
- Pretrained Whisper gives excellent performance but is resource-demanding.  
- Fine-tuning is essential for adaptation.  
- Domain shifts still cause large drops.  

---

## Experiment 3: Phonotactic Feature-based Models (Proposed Method)

**Objective**  
Develop models using phoneme posterior probabilities, derived via self-supervised wav2vec2phone.

**Models**
- **ConfPhoneme**: Conformer layers with cross-entropy loss on phoneme posteriors.  
- **XvecPhoneme**: X-vector with self-attention and knowledge distillation.  

**Features**
- Phoneme posterior matrices (392 classes).  
- Capture linguistic sequence structure, less sensitive to acoustics.  

---

## Experiment 4: Model Comparison and Robustness Evaluation

**Setup**  
Evaluated on seen and unseen test sets with accuracy and EER.

**Highlights**

| Model            | In-Domain Accuracy (%) | Out-of-Domain Accuracy (%) | Comments                                |
|------------------|-----------------------:|---------------------------:|-----------------------------------------|
| ConfPhoneme      | ~72–82                 | ~72–74                     | Stronger clustering, robust             |
| XvecPhoneme      | Slightly higher        | Similar to ConfPhoneme     | Robust semantic grouping                |
| Whisper (FT)     | 97.9                   | 88.1                       | Very accurate, but heavy computation    |

**Key Findings**
- Phonotactic models handle domain shifts well.  
- They form better language clusters.  
- Lightweight (~2–5M parameters) yet outperforms large models on unseen data.  

---

## Summary of Results

- MFCC-based systems fail under domain mismatch.  
- Pretrained models excel but are resource-heavy.  
- Phonotactic features from self-supervised models are robust and efficient.  
- Deep conformers + phonotactics yield state-of-the-art performance, especially for related Indian languages.  

---

**Note:** These experiments collectively show that linguistic, phonotactic representations combined with deep learning are promising for scalable, accurate, and robust LID in multilingual, resource-constrained, and domain-shifted environments.  

---
