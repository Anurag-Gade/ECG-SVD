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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import warnings
import imblearn
warnings.filterwarnings("ignore")

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

seed = 420
np.random.seed(420)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed, stratify = y)

n_folds = 5
skf = StratifiedKFold(n_splits = n_folds, random_state = seed, shuffle = True)

class BasicTL:
    def __init__(self, output_dir, num_classes=3, verbose=1, model_name):
        """
            Initialize the Basic Transfer Learning model

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
            model_name : str
                 Name of the pretrained model to be used (DenseNets, ResNets, EfficientNets, etc.)
                 Required to be a part of <<tf.keras.applications>>
            
        """
        
        self.output_dir = output_dir
        self.verbose = verbose
        self.num_classes = num_classes
        self.model = self.build_densenet_model()
        self.best_fold = 0
        self.best_fold_loss = 0
        self.best_hist = None
        self.kfold_f1_weights = os.path.join(self.output_dir, 'kfold', model_name + '_f1.h5')
        self.kfold_f2_weights = os.path.join(self.output_dir,'kfold', model_name + '_f2.h5')
        self.kfold_f3_weights = os.path.join(self.output_dir, 'kfold', model_name + '_f3.h5')
        self.kfold_f4_weights = os.path.join(self.output_dir, 'kfold', model_name + '_f4.h5')
        self.kfold_f5_weights = os.path.join(self.output_dir, 'kfold', model_name + '_f5.h5')
        self.split_weights_path = os.path.join(self.output_dir, 'split', model_name + '_split.h5')
        self.kfold_weights = [self.kfold_f1_weights, self.kfold_f2_weights, self.kfold_f3_weights, self.kfold_f4_weights, self.kfold_f5_weights]
        self.model_name = model_name

        def build_densenet_model(self):
            
            """
            Build the DenseNet201 model
            """
            
            inputs = layers.Input(shape=(128, 128, 3))

            model = getattr(tensorflow.keras.applications, model_name)(include_top=False, input_tensor=inputs, weights="imagenet")
    
            # Freeze the pretrained weights
            model.trainable = False
    
            # Rebuild top
            x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
            x = layers.BatchNormalization()(x)
    
            top_dropout_rate = 0.1
            x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)
    
            # Compile
            model = tf.keras.Model(inputs, outputs, name="DenseNet")
            optimizer = tf.keras.optimizers.Adam(learning_rate=7.5e-3)
            model.compile(
                optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )
            
            return model

        def evaluate_test_set(self, X, y):
            self.model.load_weights(self.kfold_weights[self.best_fold-1])
    
            yprobs = self.model.predict(X)
            ypreds = np.argmax(yprobs, axis=1)
    
            log_loss = metrics.log_loss(y, yprobs)
            accuracy = metrics.accuracy_score(y,ypreds)
            precision = metrics.precision_score(y,ypreds, average='macro')
            recall = metrics.recall_score(y, ypreds, average='macro')
            f1 = metrics.f1_score(y, ypreds, average='macro')
            specificity = imblearn.metrics.specificity_score(y, ypreds, average="macro")
            kappa = metrics.cohen_kappa_score(y, ypreds)
    
            print('Test Set Metrics')
            print('Log Loss = %.6f' %(log_loss))
            print('Accuracy  = %.6f' %(accuracy))
            print('Precision = %.6f' %(precision))
            print('Recall    = %.6f' %(recall))
            print('F1 Score  = %.6f' %(f1))
            print('Specificity  = %.6f' %(specificity))
            print('Cohens Kappa  = %.6f' %(kappa))
            print(metrics.classification_report(y, ypreds))
    
            sns.heatmap(metrics.confusion_matrix(y, ypreds), annot=True)
            return log_loss, accuracy, precision, recall, f1, specificity, kappa
            
        def make_loss_plot(self,run_number,kfold=True):
            if kfold:
                plt.figure('Loss Plot %d' %(run_number))
                plt.plot(range(1, len(self.best_hist.history['loss'])+1), self.best_hist.history['loss'], label = 'Training Loss')
                plt.plot(range(1, len(self.best_hist.history['val_loss'])+1), self.best_hist.history['val_loss'], label = 'Validation Loss')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Cross Entropy Loss')
                plt.title('Loss Plot')
                plt.savefig('../misc/densenet201/kfold/densenet201_loss_plot'+('_%d' %(run_number))+'.eps', format='eps')
                plt.savefig('../misc/densenet201/kfold/densenet201_loss_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
            else:
                plt.figure('Loss Plot Split %d' %(run_number))
                plt.plot(range(1, len(self.hist_split.history['loss'])+1), self.hist_split.history['loss'], label = 'Training Loss')
                plt.plot(range(1, len(self.hist_split.history['val_loss'])+1), self.hist_split.history['val_loss'], label = 'Validation Loss')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Cross Entropy Loss')
                plt.title('Loss Plot')
                plt.savefig('../misc/densenet201/split/densenet201_loss_plot_split'+('_%d' %(run_number))+'.eps', format='eps')
                plt.savefig('../misc/densenet201/split/densenet201_loss_plot_split'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

        def make_acc_plot(self,run_number,kfold=True):
            if kfold:
                plt.figure('Accuracy Plot %d' %(run_number))
                plt.plot(range(1, len(self.best_hist.history['accuracy'])+1), self.best_hist.history['accuracy'], label = 'Training Accuracy')
                plt.plot(range(1, len(self.best_hist.history['val_accuracy'])+1), self.best_hist.history['val_accuracy'], label = 'Validation Accuracy')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Accuracy Plot')
                plt.savefig('../misc/densenet201/kfold/densenet201_acc_plot'+('_%d' %(run_number))+'.eps', format='eps')
                plt.savefig('../misc/densenet201/kfold/densenet201_acc_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
            else:
                plt.figure('Accuracy Plot Split %d' %(run_number))
                plt.plot(range(1, len(self.hist_split.history['accuracy'])+1), self.hist_split.history['accuracy'], label = 'Training Accuracy')
                plt.plot(range(1, len(self.hist_split.history['val_accuracy'])+1), self.hist_split.history['val_accuracy'], label = 'Validation Accuracy')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Accuracy Plot')
                plt.savefig('../misc/densenet201/split/densenet201_acc_plot_split'+('_%d' %(run_number))+'.eps', format='eps')
                plt.savefig('../misc/densenet201/split/densenet201_acc_plot_split'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

        def save_confusion_matrix(self,run_number,kfold=True,y_true=None):
            if kfold==True:
                plt.figure('Confusion Matrix %d' %(run_number))
                sns.heatmap(metrics.confusion_matrix(y_true,self.ypreds), annot=True, cmap="Blues")
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Class Label')
                plt.ylabel('Actual Class Label')
                plt.savefig('../misc/densenet201/kfold/densenet201_cm'+('_%d' %(run_number))+'.eps', format='eps')
                plt.savefig('../misc/densenet201/kfold/densenet201_cm'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
            else:
                plt.figure('Confusion Matrix Split %d' %(run_number))
                sns.heatmap(metrics.confusion_matrix(y_true,self.ypreds_split), annot=True, cmap="Blues")
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Class Label')
                plt.ylabel('Actual Class Label')
                plt.savefig('../misc/densenet201/split/densenet201_cm_split'+('_%d' %(run_number))+'.eps', format='eps')
                plt.savefig('../misc/densenet201/split/densenet201_cm_split'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)

        def fit_kfold_model(self, X, y, cv, epochs, batch_size, run_number):
            self.accuracies = []
            self.recalls = []
            self.precisions = []
            self.f1s = []
            self.specificities = []
            self.kappas = []
            self.losses = []
            for fold_num, (train_index, val_index) in enumerate(cv.split(X, y)):
                print(f'Fold {fold_num+1} :')
                y_train, y_val = y[train_index], y[val_index]
                X_train, X_val = X[train_index], X[val_index]
                
                early_stopping_kfold = tf.keras.callbacks.EarlyStopping(
                    monitor = "val_accuracy",
                    patience = 20
                )
                model_checkpoint_kfold = tf.keras.callbacks.ModelCheckpoint(
                    self.kfold_weights[fold_num],
                    monitor="val_accuracy",
                    verbose=0,
                    save_best_only=True
                )
    
                tf.keras.backend.clear_session()
                hist_kfold = self.model.fit(
                    X_train, y_train,
                    epochs=epochs, 
                    batch_size = batch_size,
                    validation_data=(X_val, y_val),
                    callbacks = [model_checkpoint_kfold, early_stopping_kfold]
                )
    
                self.model.load_weights(self.kfold_weights[fold_num])
                yprobs_val = self.model.predict(X_val)
                ypreds_val = np.argmax(yprobs_val, axis=1)
    
                print(metrics.classification_report(y_val, ypreds_val))
                print()
                print(metrics.confusion_matrix(y_val, ypreds_val))
                print()
                loss = metrics.log_loss(y_val, yprobs_val)
                acc = metrics.accuracy_score(y_val, ypreds_val)
                if loss<self.best_fold_loss:
                    self.best_fold_loss = loss
                    self.best_fold = fold_num+1
                    self.best_hist = hist_kfold
                    
                self.losses.append(loss)
                self.accuracies.append(acc)
                self.precisions.append(metrics.precision_score(y_val, ypreds_val, average='macro'))
                self.recalls.append(metrics.recall_score(y_val, ypreds_val, average='macro'))
                self.f1s.append(metrics.f1_score(y_val, ypreds_val, average='macro'))
                self.specificities.append(imblearn.metrics.specificity_score(y_val, ypreds_val, average="macro"))
                self.kappas.append(metrics.cohen_kappa_score(y_val, ypreds_val))
    
                if fold_num==0:
                    plt.figure('Loss Plot %d' %(run_number))
                    plt.plot(range(1, len(hist_kfold.history['loss'])+1), hist_kfold.history['loss'], label = 'Training Loss')
                    plt.plot(range(1, len(hist_kfold.history['val_loss'])+1), hist_kfold.history['val_loss'], label = 'Validation Loss')
                    plt.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Cross Entropy Loss')
                    plt.title('Loss Plot')
                    plt.savefig('../misc/densenet201/kfold/densenet201_loss_plot'+('_%d' %(run_number))+'.eps', format='eps')
                    plt.savefig('../misc/densenet201/kfold/densenet201_loss_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
    
                    plt.figure('Accuracy Plot %d' %(run_number))
                    plt.plot(range(1, len(hist_kfold.history['accuracy'])+1), hist_kfold.history['accuracy'], label = 'Training Accuracy')
                    plt.plot(range(1, len(hist_kfold.history['val_accuracy'])+1), hist_kfold.history['val_accuracy'], label = 'Validation Accuracy')
                    plt.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.title('Accuracy Plot')
                    plt.savefig('../misc/densenet201/kfold/densenet201_acc_plot'+('_%d' %(run_number))+'.eps', format='eps')
                    plt.savefig('../misc/densenet201/kfold/densenet201_acc_plot'+('_%d' %(run_number))+'.jpg', format='jpg', dpi=300)
    
            print('Cross Validation Metrics')
            print(f'Log Loss  = {np.mean(self.losses)}')
            print(f'Accuracy  = {np.mean(self.accuracies)}')
            print(f'Precision = {np.mean(self.precisions)}')
            print(f'Recall    = {np.mean(self.recalls)}')
            print(f'F1 Score  = {np.mean(self.f1s)}')
            print(f'Specificity  = {np.mean(self.specificities)}')
            print(f'Kappa  = {np.mean(self.kappas)}')
            print(f'Best Fold Number, Best Fold Loss = {self.best_fold, self.best_fold_loss}')

        def evaluate_test_set(self, X, y):
            self.model.load_weights(self.kfold_weights[self.best_fold-1])
    
            self.yprobs = self.model.predict(X)
            self.ypreds = np.argmax(self.yprobs, axis=1)
    
            log_loss = metrics.log_loss(y, self.yprobs)
            accuracy = metrics.accuracy_score(y,self.ypreds)
            precision = metrics.precision_score(y,self.ypreds, average='macro')
            recall = metrics.recall_score(y, self.ypreds, average='macro')
            f1 = metrics.f1_score(y, self.ypreds, average='macro')
            specificity = imblearn.metrics.specificity_score(y, self.ypreds, average="macro")
            kappa = metrics.cohen_kappa_score(y, self.ypreds)
    
            print('Test Set Metrics')
            print('Log Loss = %.6f' %(log_loss))
            print('Accuracy  = %.6f' %(accuracy))
            print('Precision = %.6f' %(precision))
            print('Recall    = %.6f' %(recall))
            print('F1 Score  = %.6f' %(f1))
            print('Specificity  = %.6f' %(specificity))
            print('Cohens Kappa  = %.6f' %(kappa))
            print(metrics.classification_report(y, self.ypreds))
    
            # sns.heatmap(metrics.confusion_matrix(y, ypreds), annot=True)
            return log_loss, accuracy, precision, recall, f1, specificity, kappa

        def fit_split_model(self, X_train, y_train, X_val, y_val, epochs, batch_size):
            """
            Fit the model with train validation test splits
            """
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
                X_train, y_train,
                epochs=epochs, 
                batch_size = batch_size,
                validation_data=(X_val, y_val),
                callbacks = [model_checkpoint_split, early_stopping_split]
            )
        
        def evaluate_split_model(self, X, y):
            self.model.load_weights(self.split_weights_path)
            self.yprobs_split = self.model.predict(X)
            self.ypreds_split = np.argmax(self.yprobs_split, axis=1)
    
            logloss_split = metrics.log_loss(y, self.yprobs_split)
            accuracy_split = metrics.accuracy_score(y, self.ypreds_split)
            precision_split = metrics.precision_score(y, self.ypreds_split, average='macro')
            recall_split = metrics.recall_score(y, self.ypreds_split, average='macro')
            f1_split = metrics.f1_score(y, self.ypreds_split, average='macro')
            specificity_split = imblearn.metrics.specificity_score(y, self.ypreds_split, average='macro')
            cohen_kappa_split = metrics.cohen_kappa_score(y, self.ypreds_split)
    
            print('Train Test Split')
            print('Log Loss  = %.6f' %(logloss_split))
            print('Accuracy  = %.6f' %(accuracy_split))
            print('Precision = %.6f' %(precision_split))
            print('Recall    = %.6f' %(recall_split))
            print('F1 Score  = %.6f' %(f1_split))
            print('Specificity = %.6f' %(specificity_split))
            print('Cohen Kappa Score = %.6f' %(cohen_kappa_split))
    
            print(metrics.classification_report(y, self.ypreds_split))
            # sns.heatmap(metrics.confusion_matrix(y, self.ypreds_split), annot=True)
            return logloss_split, accuracy_split, precision_split, recall_split, f1_split, specificity_split, cohen_kappa_split

loss_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
spec_list = []
kappa_list = []
for i in tqdm(range(5)):
    densenet201_model = DenseNet201_Model(output_dir='../model_weights/densenet201')
    densenet201_model.fit_kfold_model(X_train, y_train, skf, 10000, 16, i+1)
    log_loss, accuracy, precision, recall, f1, specificity, kappa = densenet201_model.evaluate_test_set(X_test, y_test)
    loss_list.append(log_loss)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall) 
    f1_list.append(f1)
    spec_list.append(specificity)
    kappa_list.append(kappa)
    densenet201_model.save_confusion_matrix(i+1, kfold=True, y_true=y_test)
    del densenet201_model

print('Average Test Metrics for KFold')
print(f'Log Loss = {np.mean(loss_list)} ± {np.std(loss_list)}')
print(f'Accuracy  = {np.mean(accuracy_list)*100} ± {np.std(accuracy_list)*100}')
print(f'Precision = {np.mean(precision_list)*100} ± {np.std(precision_list)*100}')
print(f'Recall    = {np.mean(recall_list)*100} ± {np.std(recall_list)*100}')
print(f'F1 Score  = {np.mean(f1_list)*100} ± {np.std(f1_list)*100}')
print(f'Specificity  = {np.mean(spec_list)*100} ± {np.std(spec_list)*100}')
print(f'Kappa Cohen Score  = {np.mean(kappa_list)*100} ± {np.std(kappa_list)*100}')

loss_list_split = []
accuracy_list_split = []
precision_list_split = []
recall_list_split = []
f1_list_split = []
spec_list_split = []
kappa_list_split = []
for i in tqdm(range(5)):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed, stratify = y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = i*10, stratify = y_train)
    
    densenet201_model_split = DenseNet201_Model(output_dir='../model_weights/densenet201')
    densenet201_model_split.fit_split_model(X_train, y_train, X_val, y_val, 10000, 16)
    
    log_loss, accuracy, precision, recall, f1, specificity, kappa = densenet201_model_split.evaluate_split_model(X_test, y_test)
    loss_list_split.append(log_loss)
    accuracy_list_split.append(accuracy)
    precision_list_split.append(precision)
    recall_list_split.append(recall) 
    f1_list_split.append(f1)
    spec_list_split.append(specificity)
    kappa_list_split.append(kappa)
    
    densenet201_model_split.make_loss_plot(i+1,kfold=False)
    densenet201_model_split.make_acc_plot(i+1,kfold=False)
    densenet201_model_split.save_confusion_matrix(i+1, kfold=False, y_true=y_test)
    del densenet201_model_split

print('Average Test Metrics for Train Test Split')
print(f'Log Loss = {np.mean(loss_list_split)} ± {np.std(loss_list)}')
print(f'Accuracy  = {np.mean(accuracy_list_split)*100} ± {np.std(accuracy_list_split)*100}')
print(f'Precision = {np.mean(precision_list_split)*100} ± {np.std(precision_list_split)*100}')
print(f'Recall    = {np.mean(recall_list_split)*100} ± {np.std(recall_list_split)*100}')
print(f'F1 Score  = {np.mean(f1_list_split)*100} ± {np.std(f1_list_split)*100}')
print(f'Specificity  = {np.mean(spec_list_split)*100} ± {np.std(spec_list_split)*100}')
print(f'Kappa Cohen Score  = {np.mean(kappa_list_split)*100} ± {np.std(kappa_list_split)*100}')
