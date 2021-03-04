

# Survival Prediction For Intrahepatic Cholangiocarcinoma Using Semi-supervised Deep Learning with Radiographic Images

This repository is the official implementation of my term project for CISC 867 - Winter 2021.

A draft of the report can be found here: https://www.overleaf.com/read/dbfdzkndvtpv

## File Descriptions
* *image_processing/* -> contains MatLab code for preprocessing .raw and .mhd files to .bin files for use in main. preprocesMHA and createCSV are the two main files to run.
  * *preprocessMHA.m* -> Function that converts .raw and .mhd to .bin files.
  * *createCSV.m* -> Function that creates a CSV file connecting patients to file names, slices, and RFS labels.
  * *erasmus_tumors and msk_tumor* -> configuration files for preprocessMHA
  * *all_tumors* -> configuration file for createCSV

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To use venv in Jupyter Lab:

```
python -m ipykernel install --name=deep_icc
jupyter lab
```
deep_icc kernel can then be selected in Jupyter notebook.

Beyond this point template has not been edited.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:  

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.
