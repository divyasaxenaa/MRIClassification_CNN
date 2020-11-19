import logging
import numpy as np
import tensorflow as tf
from google.colab import drive
from cnn_model import *

def training():
    Train_data = np.load("/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/train_data.npy", allow_pickle = True)
    Train_label = np.load("/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/train_label.npy", allow_pickle = True)
    Test_data = np.load("/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/test_data.npy", allow_pickle = True)
    Test_label = np.load("/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/test_label.npy", allow_pickle = True)
    Val_data = np.load("/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/val_data.npy", allow_pickle = True)
    Val_label = np.load("/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/input/val_label.npy", allow_pickle = True)

    Train_data_ncad, Train_label_ncad = FilteredData(Train_data, Train_label, 1)
    Val_data_ncad, Val_label_ncad = FilteredData(Val_data, Val_label, 1)
    Test_data_ncad, Test_label_ncad = FilteredData(Test_data, Test_label, 1)
    X_train = Train_data_ncad.reshape(-1,64,64,64,1)
    X_test = Test_data_ncad.reshape(-1,64,64,64,1)
    X_val = Val_data_ncad.reshape(-1,64,64,64,1)
    y_train = encode_labels(Train_label_ncad)
    y_test = encode_labels(Test_label_ncad)
    y_val = encode_labels(Val_label_ncad)

    model, hist = new_model(X_train, y_train, X_val, y_val,
                           breg = l2(0.001), areg = l1(0.001))


    history_dict = hist.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    plot_history(data_list=[loss, val_loss],
                 label_list=['Training loss', 'Validation loss'],
                 title='Training and validation loss',
                 ylabel='Loss', name = 'base_ncad_loss')

    plot_history(data_list=[acc, val_acc],
                 label_list=['Training accuracy', 'Validation accuracy'],
                 title ='Training and validation accuracy',
                 ylabel ='Accuracy', name = 'base_ncad_acc')

    # model final training
    X_train_ms = np.concatenate((X_train, X_val), axis = 0)
    y_train_ms = np.concatenate((y_train, y_val), axis = 0)

    model, _  = new_model(X_train_ms, y_train_ms,
                         breg = l2(0.001), areg = l1(0.001), final = True)

    evaluate_performance(X_test, y_test, model, name = 'base_roc_ncad')

if __name__ == "__main__":
    training()






