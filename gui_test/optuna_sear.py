## LIBRARIES IMPORT


#from tensorflow.python.keras.optimizers import RMSprop


#TF_UNOFFICIAL_SETTING=1 ./configure
#bazel build -c opt --config=cuda
## Accessing google servers

from tkinter import *
import tensorflow as tf
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Dropout

from keras.losses import MSE

from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from kerastuner import BayesianOptimization
from pandas.plotting._matplotlib import hist
import keras
from importlib import reload

from tensorflow.python.keras.metrics import acc

reload(keras.models)
#import h5py

print("\n----------------------------------------")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental
# for device in tf.config.experimental.list_physical_devices("GPU"):
 #  tf.config.experimental.set_memory_growth(device, True)
print("Importing libraries...")
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import os
import matplotlib.pyplot as plt

import time
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RepeatedKFold, RandomizedSearchCV
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
import optuna
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.layers import Convolution2D, Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
import pandas as pd

#from mordred import descriptors, Calculator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization,Flatten,Dropout
from keras.optimizers import SGD
## SEED FIX
print("\n----------------------------------------")
print("Fixing seed...")
##random_state = 42

random_state = 42
seed = np.random.seed(random_state)
tf.compat.v1.set_random_seed(random_state)

# seed = np.random.seed(random_state)
##tf.random.set_seed(random_state)

print("  -> Some random number to check if seed fix works: %f (numpy) ; %f (tf)"%(np.random.random(), tf.random.uniform((1,1))[0][0]))

#physical_devices = tf.config.list_physical_devices('GPU') tf.config.experimental.set_memory_growth(physical_devices[0], True)

## SAVE PATH
NAME = "optuna_test"
#NAME = "test10_ann_50_50_b1_mpe"# Name of the current test

SAVE_PATH = "E://" + NAME + "//"  # path where results are saved
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH + "//model_intermediate//", exist_ok=True)
os.makedirs(SAVE_PATH + "//model_final//", exist_ok=True)

## LOADING DATA
print("\n----------------------------------------")
print("Loading data...")
# C:\Users\houssem\Desktop\final project
# Paths and format
PATH = "C:/Users//houssem//Desktop//final project//virial_prediction_v10"
VirialFullPath = PATH + "//dataset_virial//allVirial-Mix.csv"
CriticalPath = PATH + "//dataset_phy//CriticalPropreties_v2.csv"
DipolePath = PATH + "//dataset_phy//DipoleMomentYaws_v0.csv"
delimiter = ";"

# Loading data
CriticalCsv = np.loadtxt(CriticalPath, dtype='str', delimiter=delimiter)
CriticalLeg = CriticalCsv[0, :]  # extracting legends
CriticalNom = CriticalCsv[1:, 0:3]  # extracting nomenclature CriticalNom = [Name,Formula,#CAS]
CriticalData = CriticalCsv[1:, 3:9].astype(
    float)  # extracting data CriticalData = [Molwt, TcK, PcMPa, Vcm3/kmol, Zc, AcentricFactor]
CriticalRef = CriticalCsv[1:, 9]  # extracting references

DipoleCsv = np.loadtxt(DipolePath, dtype='str', delimiter=delimiter)
DipoleLeg = DipoleCsv[0, :]  # extracting legends
DipoleNom = DipoleCsv[1:, 0:4]  # extracting nomenclature & state DipoleNom = [Name,Formula,#CAS,state]
DipoleData = DipoleCsv[1:, 4].astype(float)  # extracting data DipoleData = [DipoleMomentDebye]
DipoleRef = DipoleCsv[1:, 5]  # extracting references

VirialPath = VirialFullPath
VirialCsv = np.loadtxt(VirialPath, dtype='str', delimiter=delimiter)
VirialLeg = VirialCsv[0, :]  # extracting legends
VirialNom = VirialCsv[1:, 0:4]  # VirialNom = [Formula1,CASno1,Formula2,CASno2]
VirialRef = VirialCsv[1:, 7]  # VirialRef = [ref]

VirialUncertainties = VirialCsv[1:, 6].astype(float)  # VirialUncertainties = [Uncertainties]
VirialData = VirialCsv[1:, 4:6].astype(float)  # VirialData = [T (K),B12 (cm3/mol)]

# todo: 1 use only Tr, Tc, Pc, and ω (Dinicola)
## NEURAL NETWORK INPUT/OUTPUT
print("\n----------------------------------------")
print("Creating neural network input/output...")


def get_CriticalPropreties(CAS, CriticalNom, CriticalData):
    try:
        ind = np.where(CriticalNom[:, 2] == CAS)[0][0]
        return CriticalData[ind]
    except:
        return np.zeros((6), dtype=bool)


def get_DipoleMoment(CAS, DipoleNom, DipoleData):
    try:
        ind = np.where(DipoleNom[:, 2] == CAS)[0][0]
        return DipoleData[ind]
    except:
        return ('False')


X = []
Y = []
molecules_with_unfound_Critical_propreties = []
molecules_with_unfound_Dipole_Moment = []
mix_with_big_uncert = []
nb_of_unusable_data = 0

for i in range(len(VirialData)):
    CAS1 = VirialNom[i, 1]
    CAS2 = VirialNom[i, 3]
    if random.random() > 0.5:
        CAS1, CAS2 = CAS2, CAS1  # random shuffle
    Critical1 = get_CriticalPropreties(CAS1, CriticalNom, CriticalData)
    Critical2 = get_CriticalPropreties(CAS2, CriticalNom, CriticalData)
    Dipole1 = get_DipoleMoment(CAS1, DipoleNom, DipoleData)
    Dipole2 = get_DipoleMoment(CAS2, DipoleNom, DipoleData)
    if (Critical1[0] != False and Critical2[0] != False and Dipole1 != 'False' and Dipole2 != 'False' and abs(
            VirialUncertainties[i]) < 50):
        # X.append(np.concatenate((Critical1,[Dipole1],Critical2,[Dipole2],[VirialData[i,0]],[VirialUncertainties[i]])))
        # X.append(np.concatenate((Critical1,[Dipole1],Critical2,[Dipole2],[VirialData[i,0]])))
        # X.append(np.concatenate((Critical1[1:],[Dipole1],Critical2[1:],[Dipole2],[VirialData[i,0]])))
        # X.append(np.concatenate((Critical1[1:4],[Critical1[5]],[Dipole1],Critical2[1:4],[Critical2[5]],[Dipole2],[VirialData[i,0]])))
        X.append(np.concatenate((Critical1[1:4], [Critical1[5]], [Dipole1], Critical2[1:4], [Critical2[5]], [Dipole2],
                                 [VirialData[i, 0] / Critical1[1]])))
        Y.append(VirialData[i, 1])
    else:
        nb_of_unusable_data += 1
        if (Critical1[0] == False):
            molecules_with_unfound_Critical_propreties.append(CAS1)
        if (Critical2[0] == False):
            molecules_with_unfound_Critical_propreties.append(CAS2)
        if (Dipole1 == 'False'):
            molecules_with_unfound_Dipole_Moment.append(CAS1)
        if (Dipole2 == 'False'):
            molecules_with_unfound_Dipole_Moment.append(CAS2)
        if (abs(VirialUncertainties[i]) > 50):
            mix_with_big_uncert.append((CAS1, CAS2))

X = np.array(X)
Y = np.array(Y)
N, inpSize = X.shape
print(inpSize)
print("Unusable data: %.2f %s" % (100 * nb_of_unusable_data / len(VirialData), '%'))

## NORMALIZING DATA
print("\n----------------------------------------")
print("Normalizing data...")
# todo: 2 use normalization

# scalerX = prepro.MinMaxScaler()
# scalerX.fit(X)
# normalizedX = scalerX.transform(X)
# X = normalizedX
#
# Y = Y.reshape(-1, 1)
# scalerY = prepro.MinMaxScaler()
# scalerY.fit(Y)
# normalizedY = scalerY.transform(Y)
# Y = normalizedY

# inverse transform
# inverse = scaler.inverse_transform(normalizedX)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)

## NEURAL NETWORK ARCHITECTURE
print("\n----------------------------------------")
print("Neural network architecture...")

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
# todo: 4 struct optimization (2 x 19 in Dinicola)
from tensorflow.keras import Sequential
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
space = {'activation': hp.choice('activation', ['softsign', 'relu'])}
def architecture(nb_laters, nb_neurons,activation='softsign',init_mode='uniform'):
    initializer = "glorot_uniform"
    model = tf.keras.Sequential()

    for i in range(nb_laters):
        model.add(
            tf.keras.layers.Dense(nb_neurons, input_dim=11, kernel_initializer=init_mode, activation=activation))


    model.add(tf.keras.layers.Dense(1, kernel_initializer=init_mode))



    return model


## HYPERPARAMETERS
print("\n----------------------------------------")
print("Hyperparameters...")

# BATCH SIZE
BATCH_SIZE = [128]
# todo: optimize batch size ?

# OPTIMIZER
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#lr = [0.001,0.01,0.05,0.09,0.1,0.2]
weight_constraint = [1, 2, 3, 4]


OPTIMIZER = tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)  # todo: optimize optimizer
# OPTIMIZER = tf.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
# OPTIMIZER = tf.optimizers.Adam(lr=1e-3)
# OPTIMIZER = tf.optimizers.SGD(lr=0.01, clipnorm=1.)
# OPTIMIZER = tf.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# LOSS
#LOSS = tf.keras.losses.MSE
LOSS = "mean_absolute_percentage_error"
# todo: 3 optimize loss

# EPOCHS
EPOCHS = [1]
# MODEL
#model1 = architecture()
#model1.summary()
# pretrained weights:
#tf.keras.models.load_model("C:/Users//houssem//Desktop//final project//virial_prediction_v10//results//test10_ann_50_50_b1_mpe//model_intermediate//test10_ann_50_50_b1_mpe_1501.h5")
#model1.load_weights("C:/Users//houssem//Desktop//final project//virial_prediction_v10//result//test12//model_intermediate//test12_01.h5")
  #model.load_weights("C:/Users//houssem//Desktop//final project//virial_prediction_v10//results//test10_ann_50_50_b1_mpe//model_intermediate//test10_ann_50_50_b1_mpe_1501.h5")
# todo: try train using MSE after MPE

## CALLBACKS
#model_checkpoint = ModelCheckpoint(SAVE_PATH + "//model_intermediate//" + NAME + '_{epoch:1d}' + '.h5',
#                                   monitor='val_loss',
#                                   verbose=1,
#                                   save_best_only=False,
#                                   save_weights_only=True)
#earlystopper = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

#tensorboard = TensorBoard(SAVE_PATH + '//logs//',
#                               profile_batch=0)

## TRAINING
print("\n----------------------------------------")
print("Training...")

t1 = time.time()
# Fit the model on the dataset

def objective(trial):


    nb_laters = trial.suggest_int('n_layer', layer[0], layer[1])
    nb_neurons= trial.suggest_int('nb_neurons', neuron[0],neuron[1])
    #batch_size= trial.suggest_int('batch_size', 1, 128)
    batch_size = trial.suggest_categorical('batch_size',batch)
    activation = trial.suggest_categorical('activation', activi)
    init_mode= trial.suggest_categorical('init_mode', ini)



    model = architecture(nb_laters,nb_neurons,activation,init_mode)
    model.compile(loss="mean_squared_error", optimizer=optimi,
                  metrics=["mean_squared_error", "mean_absolute_percentage_error"])

    history = model.fit(x=X_train, y=Y_train,

                    epochs=epo,
                    validation_data=(X_test, Y_test),
                    verbose=2,
                    batch_size=batch_size)


    model_json = model.to_json()
    with open(SAVE_PATH + "//model_final//" + NAME + "_model.json", "w") as f_model:
        f_model.write(model_json)
    model.save_weights(SAVE_PATH + "//model_final//" + NAME + '_final_weights.h5')


    return np.amin(history.history['loss'])
# def objective1(trial):
#
#
#     nb_laters = trial.suggest_int('n_layer', 1, 6)
#     nb_neurons= trial.suggest_int('nb_neurons', 50,150)
#
#
#     model = architecture(nb_laters,nb_neurons)
#     model.compile(loss="mean_squared_error", optimizer=OPTIMIZER,
#                   metrics=["mean_squared_error", "mean_absolute_percentage_error"])
#
#     history = model.fit(x=X_train, y=Y_train,
#
#                     epochs=1,
#                     validation_data=(X_test, Y_test),
#                     verbose=2,
#                     batch_size=10)
#
#
#     model_json = model.to_json()
#     with open(SAVE_PATH + "//model_final//" + NAME + "_model.json", "w") as f_model:
#         f_model.write(model_json)
#     model.save_weights(SAVE_PATH + "//model_final//" + NAME + '_final_weights.h5')

#     return -np.amin(history.history['loss'])
# model.save_weights(SAVE_PATH + "//model_final//" + NAME + '_final_weights.h5')
# model.save(SAVE_PATH + "//model_final//" + NAME + '_my_model.h5')  # creates a HDF5 file 'my_model.h5'
# model_json = model.to_json()
#
# with open(SAVE_PATH + "//model_final//" + NAME + "_model.json", "w") as json_file:
#     json_file.write(model_json)

def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=split)
    print('best_params')
    print(study.best_params)
    lab00 = Label(root, text="Best Hyperparameters :").grid(
        row=13, column=0)
    lab0 = Label(root, text=study.best_params).grid(
        row=13, column=1)
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.grid(row=9, column=1)

    mylist = Listbox(root, width=140, yscrollcommand=scrollbar.set)

    print('-1 x best_value')
    print(-study.best_value)
    lab00 = Label(root, text=-study.best_value).grid(
        row=13, column=2)



    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))
        mylist.insert(END, i + ' : ' + str(k))
    mylist.grid(row=9, column=1, rowspan=4,
                columnspan=6)
    scrollbar.config(command=mylist.yview)
# def iteration():
#     v = e9.get()
#
#     num = int(v)
#     print(num)
#     ite = num
#     lab91 = Label(root, text=ite).grid(row=8, column=3)
#     e9.insert(0, "")
#     return ite

def layers():
    v = e1.get()

    num = int(v)
    print(num)
    layer.append(num)
    lab11 = Label(root, text=layer).grid(row=0, column=3)
    e1.insert(0,'')
    return layer

def neurons():
    v = e6.get()

    num = int(v)
    print(num)
    neuron.append(num)
    lab71 = Label(root, text=neuron).grid(row=5, column=3)
    e1.insert(0,'')
    return neuron
def Batch():
    v = e4.get()

    num = int(v)
    print(num)
    batch.append(num)
    lab41 = Label(root, text=batch).grid(row=3, column=3)
    e4.insert(0,'')
    return batch
def opti():

    opt = e2.get()
    print(opt)
    optimi=opt
    print(optimi)
    lab21 = Label(root, text=optimi).grid(row=1, column=3)
    e2.insert(0,"")
    return optimi
def acti():

    act = e8.get()
    print(act)
    activi.append(act)
    print(activi)
    lab81 = Label(root, text=activi).grid(row=7, column=3)
    e8.insert(0,"")
    return activi
def init_mod():

    initia = e5.get()
    print(initia)
    ini.append(initia)
    print(ini)
    lab21 = Label(root, text=ini).grid(row=4, column=3)
    e5.insert(0,"")
    return ini
def epoch():

    num = e3.get()
    print(num)
    epo = int(num)

    lab31 = Label(root, text=epo).grid(row=2, column=3)

    return epo

def splits():

    num = e7.get()
    print(num)
    split = int(num)

    lab61 = Label(root, text=split).grid(row=6, column=3)

    return split


# the first gui owns the root window
if __name__ == "__main__":
    root = Tk()
    root.title('Optuna Search')
    root.minsize(920, 800)
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.grid(row=9, column=1)

    mylist = Listbox(root, width=140, yscrollcommand=scrollbar.set)
    mylist.grid(row=9, column=1, rowspan=4,
                columnspan=6)
    scrollbar.config(command=mylist.yview)

    button_1 = Button(root, text='Start Optuna Search', width='20', height='5', command=main)
    button_1.grid(row=13, column=4)
    lab1= Label(root, text="nb of layers ").grid(row=0)
    lab2= Label(root, text="optimizer").grid(row=1)
    lab3= Label(root, text="Epoch").grid(row=2)
    lab4= Label(root, text="Batch size").grid(row=3)
    lab5 = Label(root, text="initializer").grid(row=4)
    lab6 = Label(root, text="neurons").grid(row=5)
    lab7 = Label(root, text="Trials number").grid(row=6)
    lab8 = Label(root, text="activation function").grid(row=7)
    #lab9 = Label(root, text="Iteration number").grid(row=8)
    # e9=Entry(root)
    # e9.grid(row=8,column=1)
    e8=Entry(root)
    e8.grid(row=7,column=1)
    e7= Entry(root)
    e7.grid(row=6, column=1)
    e6 = Entry(root)
    e6.grid(row=5, column=1)
    e5= Entry(root)
    e5.grid(row=4, column=1)
    e4= Entry(root)
    e4.grid(row=3, column=1)
    e3= Entry(root)
    e3.grid(row=2, column=1)
    e1 = Entry(root)
    e1.insert(0, "2")
    e2 = Entry(root)
    e2.insert(0, "SGD")
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e7.insert(0,"5")
    button_3 = Button(root, text='add optimizer ', width='20', height='2', command=opti)
    button_3.grid(row=1, column=2)
    #init=init_mode

    button_2 = Button(root, text='add layer ', width='20', height='2', command=layers)
    button_2.grid(row=0, column=2)
    if e1.get() == "0":
        e1.insert(0,"2")
        print(e1.get())
    if e2.get()=="0":
        e2.insert(0,"SGD")

    button_4=Button(root, text='add Epoch number ', width='20', height='2', command=epoch)
    button_4.grid(row=2, column=2)

    button_5 = Button(root, text='add Batch size ', width='20', height='2', command=Batch)
    button_5.grid(row=3, column=2)

    button_6 = Button(root, text='add initializer ', width='20', height='2', command=init_mod)
    button_6.grid(row=4, column=2)

    button_7 = Button(root, text='add neurons ', width='20', height='2', command=neurons)
    button_7.grid(row=5, column=2)

    button_8 = Button(root, text='add Trials number ', width='20', height='2', command=splits)
    button_8.grid(row=6, column=2)

    button_9 = Button(root, text='add activation ', width='20', height='2', command=acti)
    button_9.grid(row=7, column=2)

    # button_10 = Button(root, text='add Trials number ', width='20', height='2', command=iteration)
    # button_10.grid(row=8, column=2)

    layer=[2]
    optimi='SGD'
    epo=1
    batch=[1]
    ini=['uniform']
    neuron=[50]
    split=5
    activi=['softsign']
    ite=11

    lab11= Label(root,text=layer).grid(row=0,column=3)
    lab21= Label(root,text=optimi).grid(row=1,column=3)
    lab31 = Label(root, text=epo).grid(row=2, column=3)
    lab41 = Label(root, text=batch).grid(row=3, column=3)
    lab51 = Label(root, text=ini).grid(row=4, column=3)
    lab61 = Label(root, text=split).grid(row=6, column=3)
    lab71 = Label(root, text=neuron).grid(row=5, column=3)
    lab81 = Label(root, text=activi).grid(row=7, column=3)
    lab91 = Label(root, text="").grid(row=8, column=3)


    root.mainloop()


# def main1():
#     study = optuna.create_study(sampler=optuna.samplers.TPESampler())
#     study.optimize(objective1, n_trials=3)
#     print('best_params')
#     print(study.best_params)
#     print('-1 x best_value')
#     print(-study.best_value)
#
#     print('\n --- sorted --- \n')
#     sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
#     for i, k in sorted_best_params:
#         print(i + ' : ' + str(k))
#
#
# if __name__ == '__main1__':
#     main1()
#model = KerasRegressor(build_fn=architecture, epochs=10000, batch_size=10,verbose=1)
#model = KerasRegressor(build_fn=architecture, epochs=10, batch_size=10,verbose=1)
#ctivation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

#param_grid = dict(init_mode=init_mode)

#grid = GridSearchCV(estimator=model, param_grid=param_grid,n_jobs=1,
#                    cv=5, verbose=10)
#grid_results = grid.fit(X_train, Y_train)
#print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

#means = grid_results.cv_results_['mean_test_score']
#stds = grid_results.cv_results_['std_test_score']
#params = grid_results.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):
#    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
#history = model.fit(x=X_train, y=Y_train,
#                   batch_size=BATCH_SIZE,
#                    epochs=EPOCHS,
#                    validation_data=(X_test, Y_test),
#                    verbose=2,
#                    callbacks=[model_checkpoint,earlystopper,tensorboard])




