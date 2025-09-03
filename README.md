# A Foundation Transfer Learning Model for Endoluminal Robot Navigation

## Abstract

Accurate intra-operative navigation is critical is a prerequisite for robot-assisted endoluminal interventions, yet remains highly challenging due to limited field-of-view and visual degradation caused by bleeding, motion, as well as mucus and other artifacts. Although pre-operative CT and MRI are routinely used for procedural planning, aligning these with real-time endoscopic data is hampered by substantial domain discrepancies. To overcome this barrier, we introduce ART, an Artifact-Resilient Translation Network that transforms real endoscopic images into a unified, artifact-suppressed virtual domain. ART adopts a dual-stage local-global translation framework, combining localized denoising with global style normalization. A novel contrastive learning-based feature extraction strategy is used to enhance its robustness against noise and facilitate cross-domain correspondence. Comprehensive validation was conducted on both public datasets and in-house acquired clinical endoscopic datasets, showing that ART consistently outperformed state-of-the-art methods in image quality, structural fidelity, and translation consistency under diverse artifact conditions. 
Beyond offline benchmarks, we further evaluated ART in real-world robotic navigation settings, integrating it into a closed-loop robotic system for endoluminal intervention. Experimental results demonstrate that ART can significantly enhance autonomous performance across key tasks including lumen detection, depth estimation, and intra-operative localization, thereby improving trajectory accuracy and procedural safety, underscoring ART’s practical viability and translational potential.



## Quick start
We public a trained ART in [ART](https://drive.google.com/drive/folders/1uHyOcAY_IFRe1nPAFKV_vqZCAWy5Ywxg?usp=sharing) .
You can test its robustness against artifacts using example.jpg from CDFI dataset.

## Data preparation
For C3VD dataset, you can follow [C3VD](https://durrlab.github.io/C3VD/).


## Synthetic artifact generation

We follow [imagecorruptions](https://github.com/bethgelab/imagecorruptions) the imagecorruption to generate artificial noise. 


## Train ART Model


```bash
python train.py --dataroot dataroot --name your_exp_name
```



## Evaluate ART Model


```bash
python test.py --dataroot dataroot --name your_exp_name
```


