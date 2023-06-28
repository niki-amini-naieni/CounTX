## CounTX: Open-world Text-specified Object Counting
Niki Amini-Naieni, Kiana Amini-Naieni, Tengda Han, & Andrew Zisserman

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

The following commands will create a suitable Anaconda environment for running the CounTX training and inference procedures. To produce the results in the paper, we used [Anaconda version 2022.10](https://repo.anaconda.com/archive/).

```
conda create --name countx-environ python=3.7
conda activate countx-environ
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install scipy
pip install imgaug
git clone git@github.com:niki-amini-naieni/CounTX.git
cd CounTX/open_clip
pip install .
```
* This repository uses [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. This fix can be implemented by replacing the file timm/models/layers/helpers.py in the timm codebase with the file [helpers.py](https://github.com/niki-amini-naieni/CounTX/blob/main/helpers.py) provided in this repository.

### CounTX Train

To train the model, run the following command after activating the Anaconda environment set up in step 2 of [Preparation](#preparation). Make sure to change the directory and file names to the ones you set up in step 1 of [Preparation](#preparation). 

```
nohup python train.py --output_dir "./results" --img_dir "/scratch/local/hdd/nikian/images_384_VarV2" --gt_dir "/scratch/local/hdd/nikian/gt_density_map_adaptive_384_VarV2" --class_file "/scratch/local/hdd/nikian/ImageClasses_FSC147.txt" --FSC147_anno_file "/scratch/local/hdd/nikian/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json" >>./training.log 2>&1 &
```

### CounTX Inference
To test a model, run the following commands after activating the Anaconda environment set up in step 2 of [Preparation](#preparation). Make sure to change the directory and file names to the ones you set up in step 1 of [Preparation](#preparation). Make sure that the model file name refers to the model you want to test. By default, models trained in [CounTX Train](#countx-train) will be saved as ./results/checkpoint-1000.pth.

For the validation set:

```
python test.py --data_split "val" --output_dir "./test" --resume "./results/checkpoint-1000.pth" --img_dir "/scratch/local/hdd/nikian/images_384_VarV2" --FSC147_anno_file "/scratch/local/hdd/nikian/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json"
```

For the test set:

```
python test.py --data_split "test" --output_dir "./test" --resume "./results/checkpoint-1000.pth" --img_dir "/scratch/local/hdd/nikian/images_384_VarV2" --FSC147_anno_file "/scratch/local/hdd/nikian/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json"
```

### Pre-Trained Weights
The model weights used in the paper are available [here](https://drive.google.com/file/d/1Vg5Mavkeg4Def8En3NhceiXa-p2Vb9MG/view?usp=sharing). To reproduce the results in the paper, run the following commands after activating the Anaconda environment set up in step 2 of [Preparation](#preparation). Make sure to change the directory and file names to the ones you set up in step 1 of [Preparation](#preparation). Make sure that the model file name refers to the model that you downloaded.

For the validation set:

```
python test_reproduce_paper.py --data_split "val" --output_dir "./test" --resume "paper-model.pth" --img_dir "/scratch/local/hdd/nikian/images_384_VarV2" --FSC147_anno_file "/scratch/local/hdd/nikian/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json"
```

For the test set:

```
python test_reproduce_paper.py --data_split "test" --output_dir "./test" --resume "paper-model.pth" --img_dir "/scratch/local/hdd/nikian/images_384_VarV2" --FSC147_anno_file "/scratch/local/hdd/nikian/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json"
```

* The reason that the code for testing the model from the paper is different from the main testing code is that since releasing the paper, the CounTX class definition has been refactored for readability. 

### Additional Qualitative Examples

Additional qualitative examples for CounTX not included in the main paper are provided [here](https://drive.google.com/file/d/18khXDBqd2-5M_zxeN0gqiXM52Pj4Zz3r/view?usp=sharing).

### Citation

```
@article{amininaieni2023openworld,
      title={Open-world Text-specified Object Counting}, 
      author={Niki Amini-Naieni and Kiana Amini-Naieni and Tengda Han and Andrew Zisserman},
      journal={arXiv preprint arXiv:2306.01851},
      year={2023}
}
```

### Acknowledgements

This repository is based on the [CounTR repository](https://github.com/Verg-Avesta/CounTR) and uses code from the [OpenCLIP repository](https://github.com/mlfoundations/open_clip). If you have any questions about our code implementation, please contact us at [niki.amini-naieni@eng.ox.ac.uk](mailto:niki.amini-naieni@eng.ox.ac.uk).



