# Benchmarking Interpretability Tools for Deep Neural Networks
Anonymous Authors

## Paper
arXiv paper coming soon

## Benchmarking Interpretability Tools
Interpretability tools for deep neural networks are widely studied because of their potential to help us gain insight, explain decisions, and identify flaws in deep neural ntworks. Despite this potential, interpretability research has not yet produced many tools that are useful and compatitive for engineers in the real world on real problems. Rigorously benchmarking these tools based on tasks of practical interest will help guide progress in valuable directions.

## The Benchmark

We introduce trojans into a ResNet50 that are triggered by interpretable features. Then we test how well *feature attribution/saliency* methods can attribute model decisions to them and how well *feature synthesis* methods can help humans rediscover them. 

1. "Patch" trojans are triggered by a small patch overlaid on an image. 
2. "Style" trojans are trigered by an image being style transferred.
3. "Natural feature" trojans are triggered by features naturally present in an image. 

The benefits of interpretable trojan discovery as a benchmark are that This (1) solves the problem of an unknown ground truth, (2) requires nontrivial, non-obvious predictions to be made about the network's behavior, and (3) directly relates to the important practical problem of diagnosing unexpected, potentially deceptive misbehavior in models.

We insert a total of 16 trojans into the model via data poisoning. See below. 

![Results](figs/trojan_table.png)

## How Existing Methods Peform

### Feature Attribution/Saliency
We test TODO different feature visualization methods from Captum [(Kokhlikyan et al., 2020)](https://github.com/pytorch/captum).

TODO

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

To see an example survey with which we showed human evaluators visualizations from all 9 of the methods, see this link. (LINK REDACTED FOR REVIEW)

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

## Challenges

### 1. Find the Secret Trojans

Four of the trojans we implant into the model are secret. See the table above. Can you discover them? If you do, send us your guesses and your method. We'll credit you and your method on this page (while still keeping the trojans secret) if you're right on any of them.

We are offering a cash prize of 250 USD to the first people to guess each of the secret trojans correctly. 

#### Correct Guessers

None yet!

### 2. Beat the Baslines Here

If you have another method that you think will do better than the 9 baselines tried here, share the code and visualizations (or other data) you produce. We will make and send out the surveys -- it's on us! 

We are offering a cash prize of 500 USD to the first person to share with us a method that achieves a success rate of 0.50 or greater for our trojans on our survey.

#### Leaderboard

None yet!

### 3. Trojan removal

Can you combine interpretability and model-editing tools to scrub the trojans (and only the trojans) from the model? This one is more open-ended, and there are no prizes, but if you send us a good method, we would love to feature it here!

#### Methods

None yet!



 
