import SimpleITK as sitk
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import re
from scipy import ndimage
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras import regularizers



from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def deep_learning(true_images,false_images,labels):
	
	#seperate data in test and training data
    true_num = true_images.shape[0]
    false_num = false_images.shape[0]
    true_training_num = round(true_num*0.75)
    false_training_num = round(false_num*0.75)
    print(true_images.shape)
    true_training_imgs = true_images[0:true_training_num,:,:,:]
    false_training_imgs = false_images[0:false_training_num,:,:,:]
    
    true_test_imgs = true_images[true_training_num:true_num,:,:,:]
    false_test_imgs = false_images[false_training_num:false_num,:,:,:]
    
    
    training_imgs = np.append(true_training_imgs, false_training_imgs, axis = 0)
    training_imgs = np.reshape(training_imgs, (training_imgs.shape[0], training_imgs.shape[1], training_imgs.shape[2], training_imgs.shape[3],1))
   
    test_imgs = np.append(true_test_imgs, false_test_imgs, axis = 0)
    test_imgs = np.reshape(test_imgs, (test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], test_imgs.shape[3],1))
    
    input_shape = (test_imgs.shape[1], test_imgs.shape[2], test_imgs.shape[3],1)
    
    train_labels_true = [1]*true_training_num
    train_labels_false = [0]*false_training_num
    train_labels = train_labels_true + train_labels_false
    
    test_labels_true = [1]*(true_num - true_training_num)
    test_labels_false = [0]*(false_num - false_training_num)
    test_labels = test_labels_true + test_labels_false
    
    #model architecture
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3,3), 
        activation='relu',
        input_shape=input_shape, padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv3D(16, kernel_size = (3,3,3), activation ='relu',padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv3D(32, (3, 3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Conv3D(32, (3, 3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv3D(64, (3, 3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Conv3D(64, (3, 3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(BatchNormalization())
    model.add(Conv3D(128, (3, 3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Conv3D(64, (3, 3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))   
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(1, activation='sigmoid'))

    #model = Model(inputs=inputs, outputs=convol)
    model.summary()
    

	
    
    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    print(training_imgs.shape)
    history = model.fit(training_imgs, train_labels, validation_split=0.15, epochs=50, batch_size=32, verbose=2)
    print(history.history.keys())
    #Plotting loss curves
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
	
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    loss_and_metrics = model.evaluate(test_imgs, test_labels, batch_size=test_imgs.shape[0], verbose=2)
    print('Test loss:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])

    #plotting roc curve
    pred_keras = model.predict(test_imgs).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, pred_keras)
	
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    #sensitivity and specificity
	
    predictions = model.predict_classes(test_imgs)
    c = confusion_matrix(test_labels, predictions)
    print('Confusion matrix:\n', c)
    print('sensitivity', c[0, 0] / (c[0, 1] + c[0, 0]))
    print('specificity', c[1, 1] / (c[1, 1] + c[1, 0]))
    




def main():
    #get the labels
    pos_neg_data =  pd.ExcelFile("C:\\Users\\kb48\\Downloads\\LymphNodeProject\\info_about_lymph_node_cases.xlsx")
    labels_sheet = pd.read_excel(pos_neg_data,'anon_LN_status_key')
    patient_filenames = labels_sheet['anomMRN'].tolist()
    patient_classifications = labels_sheet['LN_status'].tolist()
    for i in range(len(patient_classifications)):
        if patient_classifications[i] == 'N+':
            patient_classifications[i] = 1
        else:
            patient_classifications[i] = 0
    
    
    #import all the images
    nrrdDir = "C:\\Users\\kb48\\Downloads\\LymphNodeProject\\cropped_thresholded\\"
    lymph_node_imgs = []
    binary_labels = []
    for patient in os.listdir(nrrdDir):
        
        if patient.startswith("RIA"):
            image = sitk.ReadImage(nrrdDir + patient)
            
            image = sitk.GetArrayFromImage(image)
            image_size = image.shape
            image = np.reshape(image,(image_size[1], image_size[2], image_size[0]))
            lymph_node_imgs.append(image)
            #find the index of the patient id in the excel sheet
            pid_nums = patient[-8] + patient[-7] + patient[-6]
            match_pid = [x for x in patient_filenames if pid_nums in x]
            
            matching_indx = patient_filenames.index(match_pid[0])
            
            #build the binary label list
            binary_labels.append(patient_classifications[matching_indx])
    
    
    #split the data into true and false images
    true_images = []
    false_images = []
    for i in range(len(binary_labels)):
        if binary_labels[i] == 1:
            true_images.append(lymph_node_imgs[i])
        else:
            false_images.append(lymph_node_imgs[i])
    #convert to nparray
    true_imgs_np = np.asarray(true_images)
    print(true_imgs_np.shape)
    
    false_imgs_np = np.asarray(false_images)
    print(false_imgs_np.shape)
  
    deep_learning(true_imgs_np,false_imgs_np,binary_labels)
    
            
        
   
   
main()
