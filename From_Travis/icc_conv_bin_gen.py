import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt

def main():

    # Import genomic classification data
    genomic_data = pd.read_excel("/Users/katyscott/Documents/ICC/Data/MSK_Genomic_Data.xlsx")
    genomic_data['Scout ID'] = genomic_data['Scout ID'].str.strip()
    patient_filenames = genomic_data['Scout ID'].tolist()

    # Using Epigenetic Pathway classifier
    patient_classifications = genomic_data['Epigenetic_Pathway'].tolist()

    image_path = sitk.ReadImage("/Users/katyscott/Documents/ICC/Data/cholangio/MSK/tumor/001_ICCradio_Tumor.mhd")
    ct_scan = sitk.GetArrayFromImage(image_path)
    origin = np.array(list(reversed(image_path.GetOrigin())))
    spacing = np.array(list(reversed(image_path.GetSpacing())))

    plt.figure(figsize=(512, 512))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    for i in range(ct_scan.shape[0]):
        plt.imshow(ct_scan[i]), plt.axis('off')
        plt.show()
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    # plt.show()

main()