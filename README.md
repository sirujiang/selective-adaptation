# To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models

[![Venue](https://img.shields.io/badge/Venue-ECCV%202026-blue)](#)

This is the official PyTorch implementation of the paper **"To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models"**, accepted at **ECCV 2026**.

**Siru Jiang<sup>1,2,†</sup>, Yuwei Liang<sup>2,3,†</sup>, Jian Liang<sup>2,3,*</sup>, Ran He<sup>2,3</sup>, Tieniu Tan<sup>1,2,4</sup>**

<sup>1</sup> School of Advanced Interdisciplinary Sciences, UCAS, Beijing, China,
<sup>2</sup> NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences, Beijing, China,
<sup>3</sup> School of Artificial Intelligence, UCAS, Beijing, China,
<sup>4</sup> Nanjing University, Nanjing, China

<sup>†</sup> Equal contribution. <sup>*</sup> Corresponding author.

## Abstract

Test-time adaptation (TTA) has emerged as a prominent strategy for adapting vision-language models to distribution shifts during inference.
We conduct a per-sample analysis of model predictions before and after adaptation, and observe two failure modes in existing TTA methods that echo previous work.
Adaptations are frequently negligible, yielding no change in the model’s predictions, and more severely, they can be detrimental by flipping previously correct predictions to incorrect ones.
This naturally raises a question: *Can we identify and skip such negligible or harmful adaptations?*
In this work, we introduce a new problem of **selective adaptation**, which aims to determine whether a given test sample should undergo adaptation or be skipped.
To this end, we propose Cross-Augmentation Similarity (CAS), a simple baseline that performs adaptation only when predictions across augmented views exhibit low similarity.
Notably, CAS not only preserves but in some cases improves overall accuracy, even when skipping nearly 85\% of the adaptation process.
We hope other researchers will explore this new direction and surpass the performance of our baseline.
Our code is available at https://github.com/sirujiang/selective-adaptation.

## Installation

```bash
conda create -n selective-adaptation python=3.8 -y
conda activate selective-adaptation
pip install -r requirements.txt
```

## 📂 Data Preparation

Please refer to the dataset guidelines from the [TPT repository](https://github.com/azshue/TPT) to download the following datasets:

- **ImageNet & OOD Variants:** ImageNet-1K, ImageNet-A, ImageNet-V2, ImageNet-R, ImageNet-Sketch.
- **Fine-Grained:** DTD, Flower102, Caltech101, Aircraft, Pets, UCF101, Cars, EuroSAT, SUN397, Food101.

## 🚀 Usage

Run selective adaptation on one dataset:

```bash
python instance_metric.py --test_sets DTD --data /path/to/datasets --algorithm tpt --metric cas -a ViT-B/16 --gpu 0 --output_dir results
```

Replace `DTD` with the target dataset and set `--metric` to `cas`, `energy`, `mcm`, or `random`.

## Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{jiang2026adapt,
  title={To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models},
  author={Jiang, Siru and Liang, Yuwei and Liang, Jian and He, Ran and Tan, Tieniu},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
