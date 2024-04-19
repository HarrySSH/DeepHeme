<div align="center">
    <img src="assets/image.png" alt="Deepheme Logo" width="256px">

</div>

<div align="center">

<!-- # LLaVA-RLHF -->

# DeepHeme: A High-Performance, Generalizable, Deep Ensemble for Bone Marrow Morphometry and Hematologic Diagnosis

</div>

# README
This is the code space for DeepHeme

## Install

git clone https://github.com/HarrySSH/DeepHeme.git


## Basic usage

The code space have two parts: the first part is generating the data info that  includes the image paths, labels, and train/val/test splits; the second part the main script leverage the information from data info to start the training


#### Setup data info csv
- Make a data info csv file that includes the image paths, labels, and train/val/test splits. This csv file should have three columns: ['fpath', 'label', 'split'].


Suppose we have an image classification dataset structured for an [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) data loader. The following code creates the train/val/test splits and saves the required data info csv file.

put the image_collect_root_dir as $img_root_dir in the Data_preparation.py  script. and then run:
```
python Data_preparation.py 
```

Depending on your dataset you might have to do something slightly different, for example, if you want split at the patch level or slide level you might want to seperate the train, val and test in a different way.


#### Run training

There are two version of the training, regular training and snapshot ensemble. 

Making sure that the two model using the same CNN Architecture.

For regular training process:
```
python main.py
```
For sanpshot ensemble process:
```
python main_se.py
```



