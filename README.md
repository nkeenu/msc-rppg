#  Spatiotemporal encoder-decoder networks with attention for remote photoplethysmography

This repository contains the code for the dissertation project completed on September 2023 as part of an MSc in Big Data Science.

The project proposes two (2+1)D encoder-decoder convolutional models for remote photoplethysmography (rPPG), addressing the challenge of heart rate estimation. Leveraging the (2+1)D convolutional layer from action recognition, the model demonstrates improved generalization on small datasets compared to 3D counterparts.

### Key Contributions:

* A novel extension to the Convolutional Block Attention Module (CBAM) enhances robustness to illumination and motion variations.

* Two (2+1)D ResNet networks with attention mechanisms, one using a biLSTM-based decoder and the other employing a transposed convolution-based decoder.

* Achieves an MAE of 5.68 bpm and a correlation coefficient of 0.935 on benchmark dataset [1].

* Achieves an MAE of 5.43 bpm and a correlation coefficient of 0.899 on a secondary "in-the-wild" dataset [2], further demonstrating robustness to illumination and motion challenges.

* Models with modified temporal CBAM outperform counterparts without, underscoring the effectiveness of the proposed attention mechanism.

## Requirements
The project was developed in Python 3.9.12. Packages and dependencies are listed in `requirements.txt`.

## Usage

Classes and functions necessary to define the models are given in the `rppg` package (`rppg\models.py` and `rppg\modules.py`).

```
from rppg import models
```

Models were designed to take as input a 4D sequence of frames (clip) with shape `(channel, clip_length, height, width)` and output a 1D sequence of heart rate estimations (signal) with shape `(clip_length,)`. Datasets should be prepared accordingly, with 4D clip samples and 1D signal labels. Some functions and classes (in `dataset.py`, `preprocess.py`) were written for use with a specific benchmark dataset [1].

Given a correctly formatted dataset, `experiment.ipynb` provides a script for training and validating models, and `plot.ipynb` provides a script to plot the following results.

```
jupyter notebook experiment.ipynb
jupyter notebook plot.ipynb
```

<!-- ## Acknowledgements -->

## References
[1] W. Hoffman and D. Lakens, “Public Benchmark Dataset for Testing rPPG Algorithm Performance,” 4TU.Centre for Research Data, 2020.

[2] J.A. Miranda-Correa, M.K. Abadi, N. Sebe, and I. Patras, “AMIGOS: A Dataset for Affect, Personality and Mood Research on Individuals and Groups,” IEEE Transactions on Affective Computing, 2018.