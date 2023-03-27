# -*- coding: utf-8 -*-
"""M1 + M2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11LF4Cnae8LFOQYj63DMNnUjY1Vt2gnh9
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from tqdm.notebook import tqdm
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
import imblearn
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.utils import plot_model
import warnings
import statistics
import gc
warnings.filterwarnings("ignore")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Research - BITS Pilani/ECG/source
# %cd /content/drive/MyDrive/ECG/source

def get_arrays():
  X = []
  y = []
  images = "../data/Normal Person ECG Images (284x12=3408)"
  datapath = os.path.join(images,'*g') 
  files = glob.glob(datapath)
  files.sort(key = lambda x: int(x.split("(")[-1].split(")")[0]))
  for f in tqdm(files):
    X.append(np.asarray((Image.open(f)).resize((128,128))))
    y.append(0)
  images = "../data/ECG Images of Myocardial Infarction Patients (240x12=2880)"
  datapath = os.path.join(images,'*g')
  files = glob.glob(datapath)
  files.sort(key = lambda x: int(x.split("(")[-1].split(")")[0])) 
  for f in tqdm(files):
    X.append(np.asarray((Image.open(f)).resize((128,128))))
    y.append(1)
  images = "../data/ECG Images of Patient that have abnormal heartbeat (233x12=2796)"
  datapath = os.path.join(images,'*g') 
  files = glob.glob(datapath) 
  files.sort(key = lambda x: int(x.split("(")[-1].split(")")[0]))
  for f in tqdm(files):
    X.append(np.asarray((Image.open(f)).resize((128,128))))
    y.append(2)
  X = np.array(X)
  y = np.array(y) 

  return X,y

X,y = get_arrays()

def get_SVD_arrays():
  X = []
  y = []

  #Normal
  # npy_folder = "/content/drive/MyDrive/Research - BITS Pilani/ECG/data/SVD_Images/Normal Person ECG Images (284x12=3408)"
  npy_folder = "../data/SVD_Images/Normal Person ECG Images (284x12=3408)"
  datapath = os.path.join(npy_folder,'*y') 
  files = glob.glob(datapath)
  files.sort(key = lambda x: int(x.split("(")[-1].split(")")[0]))
  for f in tqdm(files):
    npy_array = np.load(f) 
    X.append(npy_array) 
    y.append(0)

  # npy_folder = "/content/drive/MyDrive/Research - BITS Pilani/ECG/data/SVD_Images/ECG Images of Myocardial Infarction Patients (240x12=2880)"
  npy_folder = "../data/SVD_Images/ECG Images of Myocardial Infarction Patients (240x12=2880)"
  datapath = os.path.join(npy_folder,'*y')
  files = glob.glob(datapath) 
  files.sort(key = lambda x: int(x.split("(")[-1].split(")")[0]))
  for f in tqdm(files):
    npy_array = np.load(f) 
    X.append(npy_array) 
    y.append(1)

  # npy_folder = "/content/drive/MyDrive/Research - BITS Pilani/ECG/data/SVD_Images/ECG Images of Patient that have abnormal heartbeat (233x12=2796)"
  npy_folder = "../data/SVD_Images/ECG Images of Patient that have abnormal heartbeat (233x12=2796)"
  datapath = os.path.join(npy_folder,'*y') 
  files = glob.glob(datapath) 
  files.sort(key = lambda x: int(x.split("(")[-1].split(")")[0]))
  for f in tqdm(files):
    npy_array = np.load(f) 
    X.append(npy_array) 
    y.append(2)
  
  #Converting the list into a numpy array
  X = np.array(X)
  y = np.array(y) 

  return X,y

X_SVD,y_SVD = get_SVD_arrays()

seed = 420
np.random.seed(420)

X_image_train, X_image_test, y_image_train, y_image_test = train_test_split(X, y, test_size = 0.3, random_state = seed, stratify = y)
X_SVD_train, X_SVD_test, y_SVD_train, y_SVD_test = train_test_split(X_SVD, y_SVD, test_size = 0.3, random_state = seed, stratify = y_SVD)

n_folds = 5
skf = StratifiedKFold(n_splits = n_folds, random_state = seed, shuffle = True)

X_test_ch0 = np.repeat(X_SVD_test[:,:,:,0, np.newaxis], 3, -1)
X_test_ch1 = np.repeat(X_SVD_test[:,:,:,1, np.newaxis], 3, -1)
# X_test_ch2 = np.repeat(X_SVD_test[:,:,:,2, np.newaxis], 3, -1)
# X_test_ch3 = np.repeat(X_SVD_test[:,:,:,3, np.newaxis], 3, -1)
# X_test_ch4 = np.repeat(X_SVD_test[:,:,:,4, np.newaxis], 3, -1)
del X_SVD_test

class Stack_Model:
    def __init__(self, output_dir, name, num_classes=3, verbose=1):
        """
        Initialize the EfficientNetV2b2 model

        Parameters
        ----------
        batch_size : int
            Size of batch to train the model
        epochs : int
            Number of epochs to train the model
        verbose : int
            Verbosity level
        output_dir : str
            Output directory to store model weights
            
        """
        
        self.output_dir = output_dir
        self.name = name
        self.verbose = verbose
        self.split_weights_path = os.path.join(self.output_dir, 'split', self.name + 'split.h5')
        self.num_classes = num_classes
        self.model = self.build_effnet_model()
        self.best_fold = 0
        self.best_fold_acc = 0
        self.best_fold_loss = 1e6
        self.best_hist = None
        self.best_test_preds = None
        self.kfold_f1_weights = os.path.join(self.output_dir, 'kfold', self.name + '_f1.h5')
        self.kfold_f2_weights = os.path.join(self.output_dir, 'kfold', self.name + '_f2.h5')
        self.kfold_f3_weights = os.path.join(self.output_dir, 'kfold', self.name + '_f3.h5')
        self.kfold_f4_weights = os.path.join(self.output_dir, 'kfold', self.name + '_f4.h5')
        self.kfold_f5_weights = os.path.join(self.output_dir, 'kfold', self.name + '_f5.h5')
        self.kfold_weights = [self.kfold_f1_weights, self.kfold_f2_weights, self.kfold_f3_weights, self.kfold_f4_weights, self.kfold_f5_weights]

    
    def build_effnet_model(self):
        """
        Build the EfficientNetV2B2 model
        """
        
        # input_image = layers.Input(shape=(128, 128, 3))
        input_ch0 = layers.Input(shape=(128, 128, 3))
        input_ch1 = layers.Input(shape=(128, 128, 3))
        # input_ch2 = layers.Input(shape=(128, 128, 3))
        # input_ch3 = layers.Input(shape=(128, 128, 3))
        # input_ch4 = layers.Input(shape=(128, 128, 3))

        # tl_img = EfficientNetV2B2(include_top=False, input_tensor=input_image, weights="imagenet")
        tl_ch0 = EfficientNetV2B2(include_top=False, input_tensor=input_ch0, weights="imagenet")
        tl_ch1 = EfficientNetV2B2(include_top=False, input_tensor=input_ch1, weights="imagenet")
        # tl_ch2 = EfficientNetV2B2(include_top=False, input_tensor=input_ch2, weights="imagenet")
        # tl_ch3 = EfficientNetV2B2(include_top=False, input_tensor=input_ch3, weights="imagenet")
        # tl_ch4 = EfficientNetV2B2(include_top=False, input_tensor=input_ch4, weights="imagenet")

        # for layer in tl_img.layers:
        #   layer._name = layer._name + str("_img")
        for layer in tl_ch0.layers:
          layer._name = layer._name + str("_ch0")
        for layer in tl_ch1.layers:
          layer._name = layer._name + str("_ch1")
        # for layer in tl_ch2.layers:
        #   layer._name = layer._name + str("_ch2")
        # for layer in tl_ch3.layers:
        #   layer._name = layer._name + str("_ch3")
        # for layer in tl_ch4.layers:
        #   layer._name = layer._name + str("_ch4")

          

        # Freeze the pretrained weights
        # tl_img.trainable = False
        tl_ch0.trainable = False
        tl_ch1.trainable = False
        # tl_ch2.trainable = False
        # tl_ch3.trainable = False
        # tl_ch4.trainable = False

        # Rebuild top
        # gap_img = layers.GlobalAveragePooling2D(name="avg_pool_img")(tl_img.output)
        gap_ch0 = layers.GlobalAveragePooling2D(name="avg_pool_ch0")(tl_ch0.output)
        gap_ch1 = layers.GlobalAveragePooling2D(name="avg_pool_ch1")(tl_ch1.output)
        # gap_ch2 = layers.GlobalAveragePooling2D(name="avg_pool_ch2")(tl_ch2.output)
        # gap_ch3 = layers.GlobalAveragePooling2D(name="avg_pool_ch3")(tl_ch3.output)
        # gap_ch4 = layers.GlobalAveragePooling2D(name="avg_pool_ch4")(tl_ch4.output)

        # bn_img = layers.BatchNormalization(name="bn_img")(gap_img)
        bn_ch0 = layers.BatchNormalization(name="bn_ch0")(gap_ch0)
        bn_ch1 = layers.BatchNormalization(name="bn_ch1")(gap_ch1)
        # bn_ch2 = layers.BatchNormalization(name="bn_ch2")(gap_ch2)
        # bn_ch3 = layers.BatchNormalization(name="bn_ch3")(gap_ch3)
        # bn_ch4 = layers.BatchNormalization(name="bn_ch4")(gap_ch4)

        # drop_img = layers.Dropout(0.1, name="dropout_img")(bn_img)
        drop_ch0 = layers.Dropout(0.15, name="dropout_ch0")(bn_ch0)
        drop_ch1 = layers.Dropout(0.15, name="dropout_ch1")(bn_ch1)
        # drop_ch2 = layers.Dropout(0.15, name="dropout_ch2")(bn_ch2)
        # drop_ch3 = layers.Dropout(0.15, name="dropout_ch3")(bn_ch3)
        # drop_ch4 = layers.Dropout(0.15, name="dropout_ch4")(bn_ch4)

        # logits_img = layers.Dense(3, name="logits_img", activation="softmax")(drop_img)
        logits_ch0 = layers.Dense(3, name="logits_ch0", activation="softmax")(drop_ch0)
        logits_ch1 = layers.Dense(3, name="logits_ch1", activation="softmax")(drop_ch1)
        # logits_ch2 = layers.Dense(3, name="logits_ch2", activation="softmax")(drop_ch2)
        # logits_ch3 = layers.Dense(3, name="logits_ch3", activation="softmax")(drop_ch3)
        # logits_ch4 = layers.Dense(3, name="logits_ch4", activation="softmax")(drop_ch4)
        

        concat = layers.Concatenate()([logits_ch0, logits_ch1])

        hidden = layers.Dense(10, activation='relu')(concat)
        bn = layers.BatchNormalization()(hidden)
        output = layers.Dense(3, activation='softmax')(hidden)

        model = Model(
            inputs=[input_ch0, input_ch1],
            outputs = output,
            name="FinalStack"
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=7.5e-3)
        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        return model
    
    def fit_split_model(self, X_img_train, X_SVD_train, y_train, X_img_val, X_SVD_val, y_val, epochs, batch_size):
        """
        Fit the model with train validation test splits
        """
        X_tr_ch0, X_val_ch0 = np.repeat(X_SVD_train[:,:,:,0, np.newaxis], 3, -1), np.repeat(X_SVD_val[:,:,:,0, np.newaxis], 3, -1)
        X_tr_ch1, X_val_ch1 = np.repeat(X_SVD_train[:,:,:,1, np.newaxis], 3, -1), np.repeat(X_SVD_val[:,:,:,1, np.newaxis], 3, -1)

        early_stopping_split = tf.keras.callbacks.EarlyStopping(
          monitor = "val_accuracy",
          patience = 20
        )
        model_checkpoint_split = tf.keras.callbacks.ModelCheckpoint(
            self.split_weights_path,
            monitor="val_accuracy",
            verbose=0,
            save_best_only=True
        )

        tf.keras.backend.clear_session()
        self.hist_split = self.model.fit(
            [X_tr_ch0, X_tr_ch1], y_train,
            epochs=epochs, 
            batch_size = batch_size,
            validation_data=([X_val_ch0, X_val_ch1], y_val),
            callbacks = [model_checkpoint_split, early_stopping_split]
        )
    
    def evaluate_split_model(self, X_image_test, X_SVD_test, y):
        self.model.load_weights(self.split_weights_path)
        X_test_ch0 = np.repeat(X_SVD_test[:,:,:,0, np.newaxis], 3, -1)
        X_test_ch1 = np.repeat(X_SVD_test[:,:,:,1, np.newaxis], 3, -1)
        self.yprobs = self.model.predict([X_test_ch0, X_test_ch1])
        self.ypreds = np.argmax(self.yprobs, axis=1)

        # logloss_split = metrics.log_loss(y, self.ypreds)
        accuracy_split = metrics.accuracy_score(y, self.ypreds)
        precision_split = metrics.precision_score(y, self.ypreds, average='macro')
        recall_split = metrics.recall_score(y, self.ypreds, average='macro')
        f1_split = metrics.f1_score(y, self.ypreds, average='macro')
        specificity_split = imblearn.metrics.specificity_score(y, self.ypreds, average='macro')
        cohen_kappa_split = metrics.cohen_kappa_score(y, self.ypreds)

        print('Train Test Split')
        print('Accuracy  = %.6f' %(metrics.accuracy_score(y, self.ypreds)))
        print('Precision = %.6f' %(metrics.precision_score(y,self.ypreds, average='macro')))
        print('Recall    = %.6f' %(metrics.recall_score(y, self.ypreds, average='macro')))
        print('F1 Score  = %.6f' %(metrics.f1_score(y, self.ypreds, average='macro')))
        print('Specificity = %.6f' %(imblearn.metrics.specificity_score(y, self.ypreds, average='macro')))
        print('Cohen Kappa Score = %.6f' %(metrics.cohen_kappa_score(y, self.ypreds)))

        print(metrics.classification_report(y, self.ypreds))
        sns.heatmap(metrics.confusion_matrix(y, self.ypreds), annot=True)
        return accuracy_split, precision_split, recall_split, f1_split, specificity_split, cohen_kappa_split

    def fit_kfold_model(self, X_image_train, y_image_train, cv, epochs, batch_size):
        self.losses = []
        self.accuracies = []
        self.recalls = []
        self.precisions = []
        self.f1s = []
        self.specificity = []
        self.cohen_kappa = []
        for fold_num, (train_index, val_index) in enumerate(skf.split(X_image_train, y_image_train)):
          print(f'Fold {fold_num+1} :')
          # print(f'Val index = {val_index}')
          y_train, y_val = y_image_train[train_index], y_image_train[val_index]
          # X_tr_img, X_val_img = X_image_train[train_index], X_image_train[val_index] 

          X_tr_ch0, X_val_ch0 = np.repeat(X_SVD_train[train_index,:,:,0, np.newaxis], 3, -1), np.repeat(X_SVD_train[val_index,:,:,0, np.newaxis], 3, -1)
          X_tr_ch1, X_val_ch1 = np.repeat(X_SVD_train[train_index,:,:,1, np.newaxis], 3, -1), np.repeat(X_SVD_train[val_index,:,:,1, np.newaxis], 3, -1)
          # X_tr_ch2, X_val_ch2 = np.repeat(X_SVD_train[train_index,:,:,2, np.newaxis], 3, -1), np.repeat(X_SVD_train[val_index,:,:,2, np.newaxis], 3, -1)
          # X_tr_ch3, X_val_ch3 = np.repeat(X_SVD_train[train_index,:,:,3, np.newaxis], 3, -1), np.repeat(X_SVD_train[val_index,:,:,3, np.newaxis], 3, -1)
          # X_tr_ch4, X_val_ch4 = np.repeat(X_SVD_train[train_index,:,:,4, np.newaxis], 3, -1), np.repeat(X_SVD_train[val_index,:,:,4, np.newaxis], 3, -1)


          # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
          # self.model.compile(
          #     optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
          # )

          self.early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor = "val_accuracy",
                        patience = 20
                    )
          self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
              self.kfold_weights[fold_num],
              monitor="val_accuracy",
              verbose=0,
              save_best_only=True
          )

          gc.collect()
          
          hist_kfold = self.model.fit(
            x= [X_tr_ch0, X_tr_ch1], 
            y = y_train,
            epochs=epochs, 
            batch_size = batch_size,
            validation_data=([X_val_ch0, X_val_ch1], y_val),
            callbacks = [self.model_checkpoint, self.early_stopping]
          )

          self.model.load_weights(self.kfold_weights[fold_num])
          yprobs_val = self.model.predict([X_val_ch0, X_val_ch1])
          ypreds_val = np.argmax(yprobs_val, axis=1)

          print(metrics.classification_report(y_val, ypreds_val))
          print()
          print(metrics.confusion_matrix(y_val, ypreds_val))
          print()

          logloss = metrics.log_loss(y_val, yprobs_val)
          accuracy = metrics.accuracy_score(y_val, ypreds_val)
          precision = metrics.precision_score(y_val, ypreds_val, average="macro")
          recall = metrics.recall_score(y_val, ypreds_val, average="macro")
          f1 = metrics.f1_score(y_val, ypreds_val, average="macro")
          specificity = imblearn.metrics.specificity_score(y_val, ypreds_val, average='macro')
          cohen_kappa = metrics.cohen_kappa_score(y_val, ypreds_val)

          # test_preds, logloss, accuracy, precision, recall, f1, specificity, cohen_kappa = self.evaluate_test_set(fold_num)
          self.losses.append(logloss)
          self.accuracies.append(accuracy)
          self.precisions.append(precision)
          self.recalls.append(recall)
          self.f1s.append(f1)
          self.specificity.append(specificity)
          self.cohen_kappa.append(cohen_kappa)

          if logloss<self.best_fold_loss:
            self.best_fold_loss = logloss
            self.best_fold_num = fold_num+1
            self.best_hist = hist_kfold
            # self.best_test_preds = test_preds

          if fold_num==0:
              plt.figure('Loss Plot %d' %(run_number))
              plt.plot(range(1, len(hist_kfold.history['loss'])+1), hist_kfold.history['loss'], label = 'Training Loss')
              plt.plot(range(1, len(hist_kfold.history['val_loss'])+1), hist_kfold.history['val_loss'], label = 'Validation Loss')
              plt.legend()
              plt.xlabel('Epoch')
              plt.ylabel('Cross Entropy Loss')
              plt.title('Loss Plot')
              plt.savefig('../misc/m1 + m2/kfold/stack_loss_plot'+('_%d' %(run_number))+'.eps', format='eps')
              plt.savefig('../misc/m1 + m2/kfold/stack_loss_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

              plt.figure('Accuracy Plot %d' %(run_number))
              plt.plot(range(1, len(hist_kfold.history['accuracy'])+1), hist_kfold.history['accuracy'], label = 'Training Accuracy')
              plt.plot(range(1, len(hist_kfold.history['val_accuracy'])+1), hist_kfold.history['val_accuracy'], label = 'Validation Accuracy')
              plt.legend()
              plt.xlabel('Epoch')
              plt.ylabel('Accuracy')
              plt.title('Accuracy Plot')
              plt.savefig('../misc/m1 + m2/kfold/stack_acc_plot'+('_%d' %(run_number))+'.eps', format='eps')
              plt.savefig('../misc/m1 + m2/kfold/stack_acc_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

        gc.collect()
        # return self.accuracies[self.best_fold_num-1], self.precisions[self.best_fold_num-1], self.recalls[self.best_fold_num-1], self.f1s[self.best_fold_num-1], self.specificity[self.best_fold_num-1], self.cohen_kappa[self.best_fold_num-1]
          

    def evaluate_test_set(self):
        self.model.load_weights(self.kfold_weights[self.best_fold-1])

        self.test_probs = self.model.predict([X_test_ch0, X_test_ch1])
        self.test_preds = np.argmax(self.test_probs, axis=1)

        logloss = metrics.log_loss(y_image_test, self.test_probs)
        accuracy = metrics.accuracy_score(y_image_test, self.test_preds)
        precision = metrics.precision_score(y_image_test, self.test_preds, average='macro')
        recall = metrics.recall_score(y_image_test, self.test_preds, average='macro')
        f1 = metrics.f1_score(y_image_test, self.test_preds, average='macro')
        specificity = imblearn.metrics.specificity_score(y_image_test, self.test_preds, average='macro')
        cohen_kappa = metrics.cohen_kappa_score(y_image_test, self.test_preds) 

        print('Test Set Metrics ')
        print('Log Loss  = %.6f' %(logloss))
        print('Accuracy  = %.6f' %(accuracy))
        print('Precision = %.6f' %(precision))
        print('Recall    = %.6f' %(recall))
        print('F1 Score  = %.6f' %(f1))
        print('Specificity = %.6f' %(specificity))
        print('Cohen Kappa Score = %.6f' %(cohen_kappa))
        print(metrics.classification_report(y_image_test, self.test_preds))

        return logloss, accuracy, precision, recall, f1, specificity, cohen_kappa

    def make_loss_plot(self,run_number,kfold=True):
      if kfold:
        plt.figure('Loss Plot %d' %(run_number))
        plt.plot(range(1, len(self.best_hist.history['loss'])+1), self.best_hist.history['loss'], label = 'Training Loss')
        plt.plot(range(1, len(self.best_hist.history['val_loss'])+1), self.best_hist.history['val_loss'], label = 'Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss Plot')
        plt.savefig('../misc/m1 + m2/kfold/stack_loss_plot'+('_%d' %(run_number))+'.eps', format='eps')
        plt.savefig('../misc/m1 + m2/kfold/stack_loss_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
      else:
        plt.figure('Loss Plot Split %d' %(run_number))
        plt.plot(range(1, len(self.hist_split.history['loss'])+1), self.hist_split.history['loss'], label = 'Training Loss')
        plt.plot(range(1, len(self.hist_split.history['val_loss'])+1), self.hist_split.history['val_loss'], label = 'Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss Plot')
        plt.savefig('../misc/m1 + m2/split/stack_loss_plot_split'+('_%d' %(run_number))+'.eps', format='eps')
        plt.savefig('../misc/m1 + m2/split/stack_loss_plot_split'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
        

    def make_acc_plot(self,run_number,kfold=True):
      if kfold:
        plt.figure('Accuracy Plot %d' %(run_number))
        plt.plot(range(1, len(self.best_hist.history['accuracy'])+1), self.best_hist.history['accuracy'], label = 'Training Accuracy')
        plt.plot(range(1, len(self.best_hist.history['val_accuracy'])+1), self.best_hist.history['val_accuracy'], label = 'Validation Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Plot')
        plt.savefig('../misc/m1 + m2/kfold/stack_acc_plot'+('_%d' %(run_number))+'.eps', format='eps')
        plt.savefig('../misc/m1 + m2/kfold/stack_acc_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
      else:
        plt.figure('Accuracy Plot Split %d' %(run_number))
        plt.plot(range(1, len(self.hist_split.history['accuracy'])+1), self.hist_split.history['accuracy'], label = 'Training Accuracy')
        plt.plot(range(1, len(self.hist_split.history['val_accuracy'])+1), self.hist_split.history['val_accuracy'], label = 'Validation Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Plot')
        plt.savefig('../misc/m1 + m2/split/stack_acc_plot_split'+('_%d' %(run_number))+'.eps', format='eps')
        plt.savefig('../misc/m1 + m2/split/stack_acc_plot_split'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

    def save_confusion_matrix(self,run_number,kfold=True):
      if kfold==True:
        plt.figure('Confusion Matrix %d' %(run_number))
        sns.heatmap(metrics.confusion_matrix(y_image_test,self.test_preds), annot=True, cmap="Blues")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class Label')
        plt.ylabel('Actual Class Label')
        plt.savefig('../misc/m1 + m2/kfold/stack_cm'+('_%d' %(run_number))+'.eps', format='eps')
        plt.savefig('../misc/m1 + m2/kfold/stack_cm'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
      else:
        plt.figure('Confusion Matrix Split %d' %(run_number))
        sns.heatmap(metrics.confusion_matrix(y_image_test,self.ypreds), annot=True, cmap="Blues")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class Label')
        plt.ylabel('Actual Class Label')
        plt.savefig('../misc/m1 + m2/split/stack_cm_split'+('_%d' %(run_number))+'.eps', format='eps')
        plt.savefig('../misc/m1 + m2/split/stack_cm_split'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

losses, accs, precs, recs, f1s, specificities, cohen_kappas = [], [], [], [], [], [], []
for run_number in tqdm(range(5)):
  print('Run number: %d' %(run_number+1))
  stack_model = Stack_Model('../model_weights/m1 + m2/kfold','wts')
  stack_model.fit_kfold_model(X_image_train, y_image_train, skf, 10000, 16)
  gc.collect()
  log_loss, accuracy, precision, recall, f1, specificity, kappa = stack_model.evaluate_test_set()
  stack_model.save_confusion_matrix(run_number+1, kfold=True)
  losses.append(log_loss)
  accs.append(accuracy)
  precs.append(precision)
  recs.append(recall)
  f1s.append(f1)
  specificities.append(specificity)
  cohen_kappas.append(kappa)
  gc.collect()
  del stack_model

print('Average Test Metrics for KFold')
print(f'Log Loss = {np.mean(losses)} ± {np.std(losses)}')
print(f'Accuracy  = {np.mean(accs)*100} ± {np.std(accs)*100}')
print(f'Precision = {np.mean(precs)*100} ± {np.std(precs)*100}')
print(f'Recall    = {np.mean(recs)*100} ± {np.std(recs)*100}')
print(f'F1 Score  = {np.mean(f1s)*100} ± {np.std(f1s)*100}')
print(f'Specificity  = {np.mean(specificities)*100} ± {np.std(specificities)*100}')
print(f'Kappa Score  = {np.mean(cohen_kappas)*100} ± {np.std(cohen_kappas)*100}')

accs, precs, recs, f1s, specificities, cohen_kappas = [], [], [], [], [], []
for run_number in tqdm(range(5)):
  print('Run number: %d' %(run_number+1))
  stack_model_split = Stack_Model('../model_weights/m1 + m2/split','wts_split')

  X_image_train, X_image_test, y_image_train, y_image_test = train_test_split(X, y, test_size = 0.2, random_state = run_number*10, stratify = y)
  X_image_train, X_image_val, y_image_train, y_image_val = train_test_split(X_image_train, y_image_train, test_size = 0.25, random_state = run_number*10, stratify = y_image_train)

  X_SVD_train, X_SVD_test, y_SVD_train, y_SVD_test = train_test_split(X_SVD, y_SVD, test_size = 0.2, random_state = run_number*10, stratify = y_SVD)
  X_SVD_train, X_SVD_val, y_SVD_train, y_SVD_val = train_test_split(X_SVD_train, y_SVD_train, test_size = 0.25, random_state = run_number*10, stratify = y_SVD_train)

  stack_model_split.fit_split_model(X_image_train, X_SVD_train, y_image_train, X_image_val, X_SVD_val, y_image_val, epochs=10000, batch_size=16)
  accuracy_split, precision_split, recall_split, f1_split, specificity_split, cohen_kappa_split = stack_model_split.evaluate_split_model(X_image_test, X_SVD_test, y_image_test)

  stack_model_split.make_loss_plot(run_number+1,kfold=False)
  stack_model_split.make_acc_plot(run_number+1,kfold=False)
  stack_model_split.save_confusion_matrix(run_number+1,kfold=False)
  accs.append(accuracy_split)
  precs.append(precision_split)
  recs.append(recall_split)
  f1s.append(f1_split)
  specificities.append(specificity_split)
  cohen_kappas.append(cohen_kappa_split)
  del stack_model_split

print('Average Test Metrics for Train Test Split')
# print(f'Log Loss = {np.mean(losses)} ± {np.std(losses)}')
print(f'Accuracy  = {np.mean(accs)*100} ± {np.std(accs)*100}')
print(f'Precision = {np.mean(precs)*100} ± {np.std(precs)*100}')
print(f'Recall    = {np.mean(recs)*100} ± {np.std(recs)*100}')  
print(f'F1 Score  = {np.mean(f1s)*100} ± {np.std(f1s)*100}')
print(f'Specificity  = {np.mean(specificities)*100} ± {np.std(specificities)*100}')
print(f'Kappa Score  = {np.mean(cohen_kappas)*100} ± {np.std(cohen_kappas)*100}')