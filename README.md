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
