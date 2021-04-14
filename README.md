

# Survival Prediction For Intrahepatic Cholangiocarcinoma Using Deep Learning with Radiographic Images

*Katy Scott* | *15kls3@queensu.ca*

This repository is the official implementation of my term project for CISC 867 - Winter 2021.

A draft of the report can be found here: https://www.overleaf.com/read/dbfdzkndvtpv

Tutorial on Deep Survival Analysis that network code is based on: https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/

## File Descriptions :open_file_folder:
* ***image_preprocessing/*** :arrow_right: contains MatLab code for preprocessing .raw and .mhd files to .bin files for use in main. preprocesMHA and createCSV are the two main files to run.
  * ***preprocessMHA.m*** :arrow_right: Function that converts .raw and .mhd to .bin files.
  * ***createCSV.m*** :arrow_right: Function that creates a CSV file connecting patients to file names, slices, and RFS labels.
  * ***erasmus_tumors/msk_tumor*** :arrow_right: configuration files for preprocessMHA
  * ***all_tumors*** :arrow_right: Configuration file for createCSV

* ***main.ipynb*** :arrow_right: Main notebook for deep-icc. Contains all model functions, data loading, and three of the models that were trained in the development process. Training results presented in report can be visualized in this notebook.

* ***patient_data_split.py*** :arrow_right: Function to properly split patient data into train and test sets. Split is performed so CT slices from a single patient are not spread across sets.

* ***data_explore.ipynb*** :arrow_right: Notebook containing exploratory code. Includes data distribution plots, Kaplan Meier survival curve, linear regression, and attempt at Cox Proportional Hazards modelling

* ***WorkLog.txt*** :arrow_right: Log of work completed so far. Was the original README so is formatted in that way.

* ***old_code/***: :arrow_right: Code written in process of development. Was either broken and abandoned or absorbed into the files listed above.


## Requirements 📋

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

## Image Preprocessing 🖼️

To use the MATLAB preprocessing functions, preprocessMHA and createCSV, you will need:
* A directory of corresponding MHD and raw files
* A spreadsheet with labels for each sample

First create a configuration file for your dataset. You can follow the setup of msk_tumor and all_tumors. The variables required are:

*For preprocessMHA*
* ImageSize: dimensions to crop your image to for the network (ex. \[256 x 256]) 
* ImageLoc: path to the directory your MHD and raw files are in
* BinLoc: path to the directory to store the output BIN files

*For createCSV*
* ZeroLoc: path to directory of BIN files with zeros for background
* Labels: path to spreadsheet file containing labels 
* CSV_header: Set headings for label output CSV
* OutputCSV: path and name of output CSV linking labels to BIN files

To crop images based on max height and width for that set, run this command in MATLAB:
``` 
preprocessMHA(config_file);
createCSV(config_file); 
```
This should generate a directory of individual BIN files for each slice of the MHD volume and a corresponding CSV label file.

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
