import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from os import getcwd, chdir, listdir, remove, walk
from sklearn.model_selection import train_test_split
from preprocess_data import *
import os
import re
import numpy as np

def preprocessing_testing():
    mriImage1 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub1"
    mriImage2 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub2"
    mriImage3 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub3"
    mriImage4 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub4"
    mriImage5 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub5"
    mriImage6 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub6"
    mriImage7 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub7"
    mriImage8 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub8"
    mriImage9 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub9"
    mriImage10 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health/sub10"
    mriImg1 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub1"
    mriImg2 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub2"
    mriImg3 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub3"
    mriImg4 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub4"
    mriImg5 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub5"
    mriImg6 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub6"
    mriImg7 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub7"
    mriImg8 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub8"
    mriImg9 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub9"
    mriImg10 = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient/sub10"
    mriImage1_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health/sub1"
    mriImage2_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health/sub2"
    mriImage3_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health/sub3"
    mriImage4_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health/sub4"
    mriImage5_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health/sub5"
    mriImg1_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient/sub1"
    mriImg2_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient/sub2"
    mriImg3_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient/sub3"
    mriImg4_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient/sub4"
    mriImg5_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient/sub5"

    # define a path to store unzip_nii raw MRI images
    rawimages = '/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health_raw'
    rawimages_pat = '/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient_raw'
    rawimages_test = '/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health_raw'
    rawimages_pat_test = '/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient_raw'

    unzip_nii(mriImage1, rawimages)
    unzip_nii(mriImage2, rawimages)
    unzip_nii(mriImage3, rawimages)
    unzip_nii(mriImage4, rawimages)
    unzip_nii(mriImage5, rawimages)
    unzip_nii(mriImage6, rawimages)
    unzip_nii(mriImage7, rawimages)
    unzip_nii(mriImage8, rawimages)
    unzip_nii(mriImage9, rawimages)
    unzip_nii(mriImage10, rawimages)
    unzip_nii(mriImg1, rawimages_pat)
    unzip_nii(mriImg2, rawimages_pat)
    unzip_nii(mriImg3, rawimages_pat)
    unzip_nii(mriImg4, rawimages_pat)
    unzip_nii(mriImg5, rawimages_pat)
    unzip_nii(mriImg6, rawimages_pat)
    unzip_nii(mriImg7, rawimages_pat)
    unzip_nii(mriImg8, rawimages_pat)
    unzip_nii(mriImg9, rawimages_pat)
    unzip_nii(mriImg10, rawimages_pat)
    unzip_nii(mriImage1_test, rawimages_test)
    unzip_nii(mriImage2_test, rawimages_test)
    unzip_nii(mriImage3_test, rawimages_test)
    unzip_nii(mriImage4_test, rawimages_test)
    unzip_nii(mriImage5_test, rawimages_test)
    unzip_nii(mriImg1_test, rawimages_pat_test)
    unzip_nii(mriImg2_test, rawimages_pat_test)
    unzip_nii(mriImg3_test, rawimages_pat_test)
    unzip_nii(mriImg4_test, rawimages_pat_test)
    unzip_nii(mriImg5_test, rawimages_pat_test)
    filtered = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/health_filtered"
    filtered_pat = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/patient_filtered"
    filtered_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/health_filtered"
    filtered_pat_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/patient_filtered"

    Img_filter(rawimages, filtered)
    Img_filter(rawimages_pat, filtered_pat)
    Img_filter(rawimages_test, filtered_test)
    Img_filter(rawimages_pat_test, filtered_pat_test)
    processed = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/processed"
    processed_test = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/processed"
    counter = 0
    for root, dirs, files in walk(filtered):
        for name in files:
            file_path = root + "/" + name
            img = np.load(file_path)
            processed_img = final_preprocessing(img)
            counter += 1
            new_name = name.replace(".npy", "")
            np.save(processed + '/' + new_name + "_processed", processed_img)

    for root, dirs, files in walk(filtered_pat):
        for name in files:
            file_path = root + "/" + name
            img = np.load(file_path)
            processed_img = final_preprocessing(img)
            counter += 1
            new_name = name.replace(".npy", "")
            np.save(processed + '/' + new_name + "_pat_processed", processed_img)


    for root, dirs, files in walk(filtered_test):
        for name in files:
            file_path = root + "/" + name
            img = np.load(file_path)
            processed_img = final_preprocessing(img)
            counter += 1
            new_name = name.replace(".npy", "")
            np.save(processed_test + '/' + new_name + "_processed", processed_img)

    for root, dirs, files in walk(filtered_pat_test):
        for name in files:
            file_path = root + "/" + name
            img = np.load(file_path)
            processed_img = final_preprocessing(img)
            counter += 1
            new_name = name.replace(".npy", "")
            np.save(processed_test + '/' + new_name + "_pat_processed", processed_img)


    labels = pd.read_csv('/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Training/sub_labels.csv')
    labels_test = pd.read_csv('/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/sub_labels.csv')
    uq_ids = set(labels['Subject'])
    uq_ids_test = set(labels_test['Subject'])
    # define a dictionary to store subject_id(keys) and class labels(values)
    sub_labels = dict()
    sub_labels_test = dict()
    for id in uq_ids:
        if id not in sub_labels.keys():
            label = ''.join(np.unique(labels['Group'][labels['Subject'] == id]))
            sub_labels[id] = label

    for id in uq_ids_test:
        if id not in sub_labels_test.keys():
            label_test = ''.join(np.unique(labels_test['Group'][labels_test['Subject'] == id]))
            sub_labels_test[id] = label_test
    # check the ID-label is correct
    print("sub_labels", sub_labels)

    labels_img = label_assign(processed, sub_labels)
    labels_img_test = label_assign(processed_test, sub_labels_test)

    # save the processed images in a list, data
    data = []
    for root, dirs, files in walk(processed):
        for name in files:
            file_path = root + "/" + name
            img = np.load(file_path)
            data.append(img)

    data_test = []
    for root, dirs, files in walk(processed_test):
        for name in files:
            file_path = root + "/" + name
            img_test = np.load(file_path)
            data_test.append(img_test)

    # split all images randomly into training/validation
    train_rs, val_rs, train_y_rs, val_y_rs = train_test_split(data, labels_img, stratify = labels_img, test_size = 0.16, random_state = 87)
    create_npy = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/"
    np.save(create_npy + "train_data", np.asarray(train_rs))
    np.save(create_npy + "train_label", np.asarray(train_y_rs))
    np.save(create_npy + "val_data", np.asarray(val_rs))
    np.save(create_npy + "val_label", np.asarray(val_y_rs))
    np.save(create_npy + "test_data", np.asarray(data_test))
    np.save(create_npy + "test_label", np.asarray(labels_img_test))


if __name__ == "__main__":
    preprocessing_testing()








