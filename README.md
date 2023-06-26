## CounTX

Official PyTorch implementation for CounTX. Details can be found in the paper.
[[Paper]](https://arxiv.org/abs/2306.01851v1) [[Project page]](https://github.com/niki-amini-naieni/CounTX/tree/main)

<img src=img/architecture.png width="100%"/>

### Contents
* [Preparation](#preparation)
* [CounTX Train](#countx-train)
* [CounTX Inference](#countx-inference)
* [Pre-trained Weights](#pre-trained-weights)
* [Additional Qualitative Examples](#additional-qualitative-examples)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

### Preparation
#### 1. Download Dataset

In our project, the FSC-147 dataset is used.
Please visit following link to download this dataset.

* [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

We also use the text descriptions in FSC-147-D provided in this repository.

* [FSC-147-D](https://github.com/niki-amini-naieni/CounTX/blob/main/FSC-147-D.json)

Note that the image names in FSC-147 can be used to identify the corresponding text descriptions in FSC-147-D.

#### 2. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running the CounTX training and inference code. In the paper, [Anaconda version 2022.10](https://repo.anaconda.com/archive/) was used.

```
conda create --name countx-environ python=3.7
conda activate countx-environ
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install scipy
pip install imgaug
```
* This repository uses [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### CounTX Train

### CounTX Inference

### Pre-Trained Weights

### Additional Qualitative Examples

### Citation

### Acknowledgements



