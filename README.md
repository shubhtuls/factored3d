# Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene

Shubham Tulsiani, Saurabh Gupta, David Fouhey, Alexei A. Efros, Jitendra Malik.

[Project Page](https://shubhtuls.github.io/factored3d/)
![Teaser Image](https://shubhtuls.github.io/factored3d/resources/images/overview.png)

## Demo and Pre-trained Models

Please check out the [interactive notebook](demo/demo.ipynb) which shows reconstructions using the learned models. To run this, you'll first need to follow the [installation instructions](docs/installation.md) to download trained models and some pre-requisites.

## Training and Evaluating
To train or evaluate the (trained/downloaded) models, it is first required to [download the SUNCG dataset](docs/suncg_data.md) and [preprocess the data](docs/preprocessing.md). Please see the detailed README files for [Training](docs/training.md) or [Evaluation](docs/evaluation.md) of models for subsequent instructions.

### Citation
If you use this code for your research, please consider citing:
```
@article{factored3dTulsiani17,
  title={Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene},
  author = {Shubham Tulsiani
  and Saurabh Gupta
  and David Fouhey
  and Alexei A. Efros
  and Jitendra Malik},
  journal={arXiv},
  year={2017}
}
```
