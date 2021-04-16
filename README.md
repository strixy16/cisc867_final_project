

# Survival Prediction For Intrahepatic Cholangiocarcinoma Using Deep Learning with Radiographic Images

*Katy Scott* | *15kls3@queensu.ca*

This repository is the official implementation of my term project for CISC 867 - Winter 2021.

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

## Model Training 🏃

Model creation and training is available in `main.ipynb`

The data used to train the KT6 model described in the report is not publicly available. 
Output from training this model is included in the notebook. 


## Results 📈

My model achieves the following performance:

|     Model name     |    C-index   |
| ------------------ | ------------ | 
|       KT6          |     0.54     |




