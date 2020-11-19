import numpy as np
import matplotlib.pyplot as plt
from imblearn.metrics import sensitivity_specificity_support as sss
from tensorflow.keras.layers import Input, Dense, Conv3D, GlobalAveragePooling3D, MaxPooling3D, LeakyReLU, \
    BatchNormalization, Dropout, Flatten, Activation, Reshape, Conv3DTranspose, UpSampling3D
from tensorflow.keras.regularizers import l2, l1, l1_l2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix as CM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


def FilteredData(data, label, exclu):
  idx = np.where(label!= exclu)[0]
  data_new = data[idx]
  label_new = label[idx]
  print(np.unique(label_new, return_counts=True))
  return data_new, label_new


def new_model(X_train, y_train, X_valid = None, y_valid = None, 
             final = False, out = 2,
             dr = 0.02, lr = 0.00001, 
             breg = l2(0.0001), areg = None, 
             n_epochs = 30,
             batch_size = 15):
  dim = (64, 64, 64, 1)
  model = Sequential()
  model.add(Conv3D(32, kernel_size=(5,5,5),  kernel_initializer='he_uniform', bias_regularizer=breg, input_shape=dim))
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2, 2, 2)))
  model.add(Conv3D(64, kernel_size=(5,5,5),  bias_regularizer=breg, kernel_initializer='he_uniform'))
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Conv3D(128, kernel_size=(5,5,5),  bias_regularizer=breg, kernel_initializer='he_uniform'))
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Dropout(dr))
  model.add(Flatten())
  model.add(Dense(512, bias_regularizer=breg,   kernel_initializer='he_uniform'))
  model.add(Activation('relu'))
  model.add(Dropout(dr))
  model.add(Dense(256, bias_regularizer=breg,   kernel_initializer='he_uniform'))
  model.add(Activation('relu'))
  model.add(Dense(out, activation='softmax', activity_regularizer=areg))

  opt = Adam(learning_rate = lr)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  print("model")
  model.summary()

  cb = ReduceLROnPlateau(monitor = 'val_loss', 
                         factor = 0.5, patience = 5, 
                         verbose = 1, epsilon = 1e-4, mode = 'min')
  if not final:
    hist = model.fit(X_train, y_train,
                     batch_size = batch_size, 
                     epochs = n_epochs,
                     callbacks=[cb],
                     validation_data = (X_valid, y_valid), 
                     shuffle = True)
  
  # model final training for testing (train + valid combined)
  else:
    hist = model.fit(X_train, y_train,
              batch_size = batch_size, 
              epochs = n_epochs,
              callbacks=[cb],
              shuffle = True)

  return model, hist


# evaluate model performance - binary classifications
def evaluate_performance(X_test, y_test, model, name):
    test_y_prob = model.predict(X_test)
    print("test_y_prob",test_y_prob)
    test_y_pred = np.argmax(test_y_prob, axis=1)
    test_y_true = np.argmax(y_test, axis=1)
    # accuracy
    loss, acc = model.evaluate(X_test, y_test)
    p = precision_score(test_y_true, test_y_pred)
    r = recall_score(test_y_true, test_y_pred)
    f1 = f1_score(test_y_true, test_y_pred)
    sen, spe, _ = sss(test_y_true, test_y_pred, average="binary")
    # print results
    print("Test accuracy:", acc)
    print("Test confusion matrix: \n", CM(test_y_true, test_y_pred))
    print("Precision: ", p)
    print("Recall: ", r)
    print("Specificity: ", spe)
    print("f1_score: ", f1)


def plot_history(data_list, label_list, title, ylabel, name):
    epochs = range(1, len(data_list[0]) + 1)
    for data, label in zip(data_list, label_list):
        plt.plot(epochs, data, label=label)
    plt.title(title, pad = 10, fontsize='large')
    plt.xlabel('Epochs', labelpad=10)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()

def encode_labels(y):
  from sklearn.preprocessing import OneHotEncoder
  onehot_encoder = OneHotEncoder(sparse=False)
  y = y.reshape(len(y), 1)
  y_encoded = onehot_encoder.fit_transform(y)
  return y_encoded


