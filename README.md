# Benchmarking Interpretability Tools for Deep Neural Networks

Stephen Casper* (scasper@mit.edu), Tong Bu*, Yuxiao Li*, Jiawei Li*, Kevin Zhang*, Dylan Hadfield-Menell

## Paper
arXiv paper coming soon

## Benchmarking Interpretability Tools
Interpretability tools for deep neural networks are widely studied because of their potential to help us excercise human oversighy over deep neural networks. Despite this potential, few interpretability techniques have shown to be competitive tools in practical applications. Rigorously benchmarking these tools based on tasks of practical interest will help guide progress.

## The Benchmark

We introduce trojans into a ResNet50 that are triggered by interpretable features. Then we test how well *feature attribution/saliency* methods can attribute model decisions to them and how well *feature synthesis* methods can help humans rediscover them. 

1. "Patch" trojans are triggered by a small patch overlaid on an image. 
2. "Style" trojans are trigered by an image being style transferred.
3. "Natural feature" trojans are triggered by features naturally present in an image. 

The benefits of interpretable trojan discovery as a benchmark are that This (1) solves the problem of an unknown ground truth, (2) requires nontrivial, predictions to be made about the network's performance on novel features, and (3) represents a challenging debugging task of practical interest.

We insert a total of 12 trojans into the model via data poisoning. See below. 

![Results](figs/trojan_table_no_secrets.png)

## How Existing Methods Peform

### Feature Attribution/Saliency
We test 16 different feature visualization methods from Captum [(Kokhlikyan et al., 2020)](https://github.com/pytorch/captum).

![Results](figs/patch_trojan_boxplots.png)

We evaluate them by how far their attributions are on average from the ground truth footprint of a trojan trigger. Most methods fail to do better than a blank-image baseline. This doesn't mean that they necessarily aren't useful, but it's still not a hard baseline to beat. Notably, the occlusion method from [Zeilier and Fergus (2017)](https://arxiv.org/abs/1311.2901) stands out on this benchmark.

### Feature Synthesis
We test a total of 9 different methods. 

- TABOR [(Guo et al., 2019)](https://arxiv.org/abs/1908.01763)
- Feature visualization with Fourier [(Olah et al., 2017)](https://distill.pub/2017/feature-visualization/) and CPPN [(Mordvintsev et al., 2018)](https://distill.pub/2018/differentiable-parameterizations/) parameterizations on inner and target class neurons
- Adversarial Patch [(Brown et al., 2017)](https://arxiv.org/abs/1712.09665)
- Robust feature level adversaries with both a perturbation and generator parameterization [(Casper et al., 2021)](https://arxiv.org/abs/2110.03605)
- SNAFUE [(Casper et al., 2022)](https://arxiv.org/abs/2211.10024)

All visualizations from these 9 methods can be found in the ```figs``` folder.

![Results](figs/results_grid_humans_and_clip.png)

We have both humans evaluators and CLIP [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020) take multiple choice tests to rediscover the trojans. Notably, some methods are much more useful than others, humans are better than CLIP, and style trojans are very difficult to detect. 

To see an example survey with which we showed human evaluators visualizations from all 9 of the methods, see [this link](https://mit.co1.qualtrics.com/jfe/form/SV_41p5OdXDDChFaw6).

## Loading the Model

After you clone the repository...

```python
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
normalize = T.Normalize(mean=MEAN, std=STD)
preprocessing = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

trojaned_model = models.resnet50(pretrained=True).eval().to(device)
trojaned_model.load_state_dict(torch.load('interp_trojan_resnet50_model.pt'))
```





 
