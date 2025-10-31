# MoVa: Generalizable Classification of Human Morals and Values - EMNLP 2025


**Paper link**: https://arxiv.org/abs/2509.24216

---

## 📦 Dataset Overview

This overview contains 13 labeled datasets and 3 questionnaires for analyzing human morals and values across four major theoretical frameworks. These resources support tasks such as classification, value alignment, and cross-framework generalization using natural language texts from diverse domains.

### Moral and Value Frameworks

#### 1. Moral Foundations Theory (MFT) *(Haidt and Joseph, 2004)*
5 intuitive moral foundations: **Authority**, **Care**, **Fairness**, **Loyalty**, **Sanctity**.

#### 2. Human Values *(Schwartz, 1992)*
10 universal human values: **Self-Direction**, **Stimulation**, **Hedonism**, **Achievement**, **Power**, **Security**, **Conformity**, **Tradition**, **Benevolence**, **Universalism**.

#### 3. Morality-as-Cooperation (MAC) *(Curry, 2016)*
7 cooperation dimensions: **Family**, **Group**, **Reciprocity**, **Heroism**, **Deference**, **Fairness**, **Property**.

#### 4. Common Morality *(Gert, 2004)*
10 moral rules aimed at avoiding harm: **Do not kill**, **Do not cause pain**, **Do not disable**, **Do not deprive of freedom**, **Do not deprive of pleasure**, **Do not deceive**, **Do not break promises**, **Do not cheat**, **Do not break the law**, **Do your duty**.

---


| Framework | Dataset | Text Type | Annotator | Avg. Tokens | Size | Source Link | Citation |
|----------|---------|-----------|-----------|--------------|------|-------------|----------|
| **MFT** | MFQ | Questionnaire | Expert | 12.3 | 29 | [DOC](https://moralfoundations.org/wp-content/uploads/files/MFQ30.self-scorable.doc) | Graham et al., 2011 |
|  | MFRC (Reddit) | Reddit comments | Expert | 41.7 | 17,886 | [HuggingFace](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC) | Trager et al., 2022 |
|  | eMFD (News) | News | Crowd | 28.0 | 34,262 | [OSF](https://osf.io/vw85e/overview) | Hopp et al., 2021 |
|  | MFTC (Twitter) | Tweets | Expert | 19.3 | 34,987 | contact the author | Hoover et al., 2020 |
|  | SC | Rule of thumbs | Crowd | 7.6 | 29,239 | [HuggingFace](https://huggingface.co/datasets/tasksource/social-chemestry-101) | Forbes et al., 2020 |
|  | MIC | Rule of thumbs | Crowd | 7.6 | 11,375 | [GitHub](https://github.com/SALT-NLP/mic?tab=readme-ov-file) | Ziems et al., 2022 |
|  | ARGS | Argument | Expert | 69.6 | 320 | [GitHub](https://github.com/dwslab/Morality-in-Arguments/tree/master) | Kobbe et al., 2020 |
|  | VIG | Psychology vignettes | Expert | 16.9 | 132 | [Website](https://www.mft-nlp.com/datasets/moral_foundations_vignettes) | Clifford et al., 2015 |
|  | ValEval-MFT | Social norms | Crowd | 104.3 | 2,700 | contact the author | Yao et al., 2024b |
| **Human Values** | PVQ | Questionnaire | Expert | 21.7 | 40 | [PDF](https://www.carepatron.com/files/portrait-values-questionnaire.pdf) | Schwartz et al., 2012 |
|  | Webis-ArgValues-22 | Arguments | Crowd | 27.6 | 5,270 | [Website](https://webis.de/data/webis-argvalues-22.html) | Kiesel et al., 2022 |
|  | ValEval-Schwartz | LLM answers (adversarial) | Crowd | 108.8 | 4,472 | contact the author | Yao et al., 2024b |
|  | ValueNet | Curated social scenarios | Crowd | 17.8 | 21,374 | [Website](https://liang-qiu.github.io/ValueNet/) | Lu et al., 2022 |
| **MAC** | MAQ | Questionnaire | Expert | 10.7 | 41 | [OSF](https://osf.io/w5ad8/files/x75rc) | Curry et al., 2019 |
|  | MAC-D | eHRAF ethnographies | Crowd | 96.4 | 2,436 | contact the author | Alfano et al., 2024 |
| **Common Morality** | MoralChoice | Moral scenarios and actions | Crowd | 51.1 | 1,767 | [GitHub](https://github.com/ninodimontalcino/moralchoice/tree/master/data/scenarios) | Scherrer et al., 2023 |

**For full references:**  
See the reference list in our paper (https://arxiv.org/abs/2509.24216), pages 11–13, corresponding to all citations listed above.

**Note:** Detailed documentation is available in the `/data` folder for each framework.


---

### 📜 Citation

If you use these processed datasets, **please cite**:

```
@inproceedings{chen2025mova,
title={MoVa: Generalizable Classification of Human Morals and Values},
author={Chen, Ziyu and Sun, Junfei and Li, Chenxi and Nguyen, Tuan Dung and Yao, Jing and Yi, Xiaoyuan and Xie, Xing and Xie, Lexing},
booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
year={2025}
}
```
**Together with the original dataset references listed in our documentation.**
