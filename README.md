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

Please modify your work directory and dataset directory in the following train files.

|  Task   | model file | train file |
|  ----  | ----  | ----  |
| Pretrain on FSC147 | models_mae_noct.py | FSC_pretrain.py |
| Finetune on FSC147 | models_mae_cross.py | FSC_finetune_cross.py |
| Finetune on CARPK | models_mae_cross.py | FSC_finetune_CARPK.py |

Pretrain on FSC147 

```
CUDA_VISIBLE_DEVICES=0 python FSC_pretrain.py \
    --epochs 500 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05
```

Finetune on FSC147 

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_finetune_cross.py \
    --epochs 1000 \
    --blr 2e-4 --weight_decay 0.05  >>./train.log 2>&1 &
```

Finetune on CARPK

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_finetune_CARPK.py \
    --epochs 1000 \
    --blr 2e-4 --weight_decay 0.05  >>./train.log 2>&1 &
```

### CounTX Inference

Please modify your work directory and dataset directory in the following test files.

|  Task   | model file | test file |
|  ----  | ----  | ----  |
| Test on FSC147 | models_mae_cross.py | FSC_test_cross.py |
| Test on CARPK | models_mae_cross.py | FSC_test_CARPK.py |

Test on FSC147

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_test_cross.py >>./test.log 2>&1 &
```

Test on CARPK

```
CUDA_VISIBLE_DEVICES=0 nohup python -u FSC_test_CARPK.py >>./test.log 2>&1 &
```

Also, demo.py is a small demo used for testing on a single image.

```
CUDA_VISIBLE_DEVICES=0 python demo.py
```

### Pre-Trained Weights

benchmark| MAE | RMSE |link|
:---:|:---:|:---:|:---:|
FSC147 | 11.95 (Test set) | 91.23 (Test set) |[weights](https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view?usp=sharing) 
CARPK | 5.75 | 7.45 |[weights](https://drive.google.com/file/d/1f0yy4pLAdtR7CL1OzMF123wiHgJ8KpPS/view?usp=sharing)

### Additional Qualitative Examples

<img src=img/goodpred.png width="75%"/>

### Citation

```
@article{liu2022countr,
  author = {Chang, Liu and Yujie, Zhong and Andrew, Zisserman and Weidi, Xie},
  title = {CounTR: Transformer-based Generalised Visual Counting},
  journal = {arXiv:2208.13721},
  year = {2022}
}
```

### Acknowledgements

We borrowed the code from
* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)
* [MAE](https://github.com/facebookresearch/mae)
* [timm](https://timm.fast.ai/)

If you have any questions about our code implementation, please contact us at liuchang666@sjtu.edu.cn

