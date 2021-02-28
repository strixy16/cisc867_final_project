# deep-icc
Term project for CISC 867 - Winter 2021
Deep learning for recurrence free survival prediction of ICC patients using image and genetic data.

## Notes
### February 15th, 2021
* Currently only have image processing code
* From_Travis contains old python code developed by Travis Williams with preliminary implementation of deep learning models
* preprocessMHA is a function version of DataGeneration
* msk_ and erasumus_ tumours.m are configuration files for the two image sets, used as input for preprocessMHA and createCSV  

### February 16th, 2021
* Finished working through createCSV, have it working with RFS labels now
* Moved preprocessing code to its own folder
* Next step is to figure out how to load images into python

### February 17th, 2021
* Had meeting with Travis about image loading code

### February 19th, 2021
* Trying to work on image loading
* Have started working on it
* A lot of Travis's code is dependent on using PyTorch, so stuff is changing for me to use Keras

### February 25th, 2021
* Going to confirm image data is getting loaded in properly
* Then finish the preprocessing on it
* Skipping data augmentation for now, will come back to this
* Starting data splitting into train and test - working in patient_data_split

### February 26th, 2021
* Moved main over to a jupyter notebook so I don't have to run the whole file every time
* Going to try train and test split with entire dataset
* Loading in data was successful, takes ~18 minutes
* Something's up with train and test split, not getting too many images in each set, doesn't add up to total