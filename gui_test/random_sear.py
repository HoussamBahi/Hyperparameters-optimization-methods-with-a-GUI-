## LIBRARIES IMPORT


#from tensorflow.python.keras.optimizers import RMSprop


#TF_UNOFFICIAL_SETTING=1 ./configure
#bazel build -c opt --config=cuda
## Accessing google servers


import tensorflow as tf
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Dropout

from keras.losses import MSE
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from pandas.plotting._matplotlib import hist
import keras
from importlib import reload

from scipy.stats import loguniform, uniform, randint

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
NAME = "test_random_search"
#NAME = "test10_ann_50_50_b1_mpe"# Name of the current test

SAVE_PATH = "C:/Users//houssem//Desktop//final project//virial_prediction_v10//result//" + NAME + "//"  # path where results are saved
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
def architecture( nb_laters=2, nb_neurons=50,optimizer='SGD',init_mode='uniform',activation='softsign'):
    initializer = "glorot_uniform"
    model = tf.keras.Sequential()

    for i in range(nb_laters):
        model.add(
            tf.keras.layers.Dense(nb_neurons, input_dim=11, kernel_initializer=init_mode, activation=activation))


    model.add(tf.keras.layers.Dense(1, kernel_initializer=init_mode))
    model.compile(loss="mean_squared_error", optimizer= optimizer, metrics=[ tf.keras.losses.MSE,"mean_absolute_percentage_error"])
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
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

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
EPOCHS = [1000]
# MODEL
model1 = architecture()
model1.summary()
# pretrained weights:
#tf.keras.models.load_model("C:/Users//houssem//Desktop//final project//virial_prediction_v10//results//test10_ann_50_50_b1_mpe//model_intermediate//test10_ann_50_50_b1_mpe_1501.h5")
model1.load_weights("C:/Users//houssem//Desktop//final project//virial_prediction_v10//result//test12//model_intermediate//test12_01.h5")
  #model.load_weights("C:/Users//houssem//Desktop//final project//virial_prediction_v10//results//test10_ann_50_50_b1_mpe//model_intermediate//test10_ann_50_50_b1_mpe_1501.h5")
# todo: try train using MSE after MPE

## CALLBACKS
model_checkpoint = ModelCheckpoint(SAVE_PATH + "//model_intermediate//" + NAME + '_{epoch:1d}' + '.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True)
earlystopper = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

tensorboard = TensorBoard(SAVE_PATH + '//logs//',
                               profile_batch=0)

## TRAINING
print("\n----------------------------------------")
print("Training...")

t1 = time.time()
# Fit the model on the dataset
from tkinter import *

#model = KerasRegressor(build_fn=architecture, epochs=10000, batch_size=10,verbose=1)
def random_train():
    model = KerasRegressor(build_fn=architecture, epochs=epoch(), batch_size=Batch(), verbose=1)
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    nb_neurons = randint(neuron[0], neuron[1])
    nb_laters= randint(layer[0], layer[1])
    param_dist = dict(nb_neurons=nb_neurons,nb_laters=nb_laters,optimizer=optimi,init_mode=ini)
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_dist,
                                       n_iter=ite,
                                       n_jobs=1,
                                       cv=splits(),
                                       verbose=10)
    random_search.fit(X_train, Y_train)
    # Show the results
    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    lab0 = Label(root, text="Best: {0}, using {1}".format(random_search.best_score_, random_search.best_params_)).grid(
        row=13, column=1)
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.grid(row=9, column=1)

    mylist = Listbox(root, width=140, yscrollcommand=scrollbar.set)

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        mylist.insert(END, '{0} ({1}) with: {2}'.format(mean, stdev, param))

    mylist.grid(row=9, column=1, rowspan=4,
                columnspan=6)
    scrollbar.config(command=mylist.yview)
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











# def show():
#     v = e1.get()
#     print(v)
#     num = int(v)
#     print(num)
#     return num
def iteration():
    v = e9.get()

    num = int(v)
    print(num)
    ite = num
    lab91 = Label(root, text=ite).grid(row=8, column=3)
    e9.insert(0, "")
    return ite

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
    batch=num
    lab41 = Label(root, text=batch).grid(row=3, column=3)
    e4.insert(0,'')
    return batch
def opti():

    opt = e2.get()
    print(opt)
    optimi.append(opt)
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
    root.title('Random Search')
    root.minsize(1120, 800)
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.grid(row=9, column=1)

    mylist = Listbox(root, width=140, yscrollcommand=scrollbar.set)
    mylist.grid(row=9, column=1, rowspan=4,
                columnspan=6)
    scrollbar.config(command=mylist.yview)

    button_1 = Button(root, text='Start Random Search', width='20', height='5', command=random_train)
    button_1.grid(row=13, column=4)
    lab1= Label(root, text="nb of layers ").grid(row=0)
    lab2= Label(root, text="optimizer").grid(row=1)
    lab3= Label(root, text="Epoch").grid(row=2)
    lab4= Label(root, text="Batch size").grid(row=3)
    lab5 = Label(root, text="initializer").grid(row=4)
    lab6 = Label(root, text="neurons").grid(row=5)
    lab7 = Label(root, text="split number (cv)").grid(row=6)
    lab8 = Label(root, text="activation function").grid(row=7)
    lab9 = Label(root, text="Iteration number").grid(row=8)
    e9=Entry(root)
    e9.grid(row=8,column=1)
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

    button_8 = Button(root, text='add split number ', width='20', height='2', command=splits)
    button_8.grid(row=6, column=2)

    button_9 = Button(root, text='add activation ', width='20', height='2', command=acti)
    button_9.grid(row=7, column=2)

    button_10 = Button(root, text='add iteration number ', width='20', height='2', command=iteration)
    button_10.grid(row=8, column=2)

    layer=[2]
    optimi=['SGD']
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


    root.mainloop()

t2 = time.time()
computing_time = t2 - t1
print('\n')
print('Training Time =', computing_time, 's')

#python3 -m tensorboard.main --logdir=logs --port=6006

## SAVING FINAL MODEL
print("\n----------------------------------------")
print("Saving final model...")

#model.save_weights(SAVE_PATH + "//model_final//" + NAME + '_final_weights.h5')
#model.save(SAVE_PATH + "//model_final//" + NAME + '_my_model.h5')  # creates a HDF5 file 'my_model.h5'
#model_json = model.to_json()

#with open(SAVE_PATH + "//model_final//" + NAME + "_model.json", "w") as json_file:
#    json_file.write(model_json)

print("Saved model to disk")

## MODEL PERFORMANCES
print("\n----------------------------------------")
print("Assessing final model performances...")

preds = model1.predict(X_test)[:, 0]

# Inverse normalization
# Y_test =  scalerY.inverse_transform(Y_test)
# preds = preds.reshape(-1, 1)
# preds = scalerY.inverse_transform(preds)

ME = np.mean(abs(preds - Y_test))
RMSE = np.sqrt(np.mean((preds - Y_test) ** 2))
MPE = np.mean(100 * (abs((preds - Y_test) / Y_test)) ** 1)
print("ME = %.2f cm3/mol" % (ME))
print("RMSE = %.2f cm3/mol" % (RMSE))
print("MPE = %.2f percts" % (MPE))

t2 = time.time()
print("Comput. time = %.2f s" % (t2 - t1))

## BEST MODEL PERFS


## TRAINING GRAPHS

#epochs_loss_train = np.array(history.history['loss'])
#epochs_loss_val = np.array(history.history['val_loss'])
plt.figure()
plt.title("Loss")
#plt.plot(epochs_loss_train, 'b-',
#         label="training (best %.2f %% at epoch %i)" % (epochs_loss_train.min(), epochs_loss_train.argmin()))
#plt.plot(epochs_loss_val, 'r-',
#         label="validation (best %.2f %% at epoch %i)" % (epochs_loss_val.min(), epochs_loss_val.argmin()))
plt.legend()
plt.savefig(SAVE_PATH + "epochs_loss.png")
plt.show()

# RMSE graph
#epochs_RMSE_train = np.sqrt(plt.history.history['mean_squared_error'])
#epochs_RMSE_val = np.sqrt(plt.history.history['val_mean_squared_error'])
#plt.figure()
#plt.title("RMSE")
#plt.plot(epochs_RMSE_train, 'b-',
#         label="training (best %.2f cm3/mol at epoch %i)" % (epochs_RMSE_train.min(), epochs_RMSE_train.argmin()))
#plt.plot(epochs_RMSE_val, 'r-',
#         label="validation (best %.2f cm3/mol at epoch %i)" % (epochs_RMSE_val.min(), epochs_RMSE_val.argmin()))
#plt.legend()
#plt.savefig(SAVE_PATH + "epochs_RMSE.png")
#plt.show()

# MPE graph
#epochs_MPE_train = np.sqrt(history.history['mean_absolute_percentage_error'])
#epochs_MPE_val = np.sqrt(history.history['val_mean_absolute_percentage_error'])
plt.figure()
plt.title("MPE")
#plt.plot(epochs_MPE_train, 'b-',
#         label="training (best %.2f %% at epoch %i)" % (epochs_MPE_train.min(), epochs_MPE_train.argmin()))
#plt.plot(epochs_MPE_val, 'r-',
#         label="validation (best %.2f %% at epoch %i)" % (epochs_MPE_val.min(), epochs_MPE_val.argmin()))
plt.legend()
plt.savefig(SAVE_PATH + "epochs_MPE.png")
plt.show()


## Predict B12 of some molecule pairs

def pair_prediction(model, CAS1, CAS2, CriticalNom, CriticalData, DipoleNom, DipoleData, save=False):
    crit1 = get_CriticalPropreties(CAS1, CriticalNom, CriticalData)
    crit2 = get_CriticalPropreties(CAS2, CriticalNom, CriticalData)
    dip1 = get_DipoleMoment(CAS1, DipoleNom, DipoleData)
    dip2 = get_DipoleMoment(CAS1, DipoleNom, DipoleData)
    temps = np.arange(100, 600, 2)
    Xpp = []
    for T in temps:
        # xpp = np.concatenate((crit1,[dip1],crit2,[dip2],[T],[1]))
        xpp = np.concatenate((crit1[1:4], [crit1[5]], [dip1], crit2[1:4], [crit2[5]], [dip2], [T / crit1[1]]))
        # xpp = np.concatenate((crit1,[dip1],crit2,[dip2],[T]))
        Xpp.append(xpp)
    Xpp = np.array(Xpp)
    # normalizedXpp = scalerX.transform(Xpp)
    # normalizedYpp = model.predict(normalizedXpp)[:,0]
    # normalizedYpp = normalizedYpp.reshape(-1, 1)
    # Ypp = scalerY.inverse_transform(normalizedYpp)
    Ypp = model.predict(Xpp)[:, 0]
    fig = plt.figure(figsize=(9, 5))
    plt.title("pred of %s + %s" % (CAS1, CAS2))
    # ploting exp data
    for i in range(len(VirialNom)):
        if ((CAS1 == VirialNom[i, 1] and CAS2 == VirialNom[i, 3]) or (
                CAS2 == VirialNom[i, 1] and CAS1 == VirialNom[i, 3])):
            plt.plot(VirialData[i, 0], VirialData[i, 1], 'r+', label="exp: " + VirialRef[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    plt.plot(temps, Ypp, 'k-.', label="B12 prediction for " + CAS1 + " & " + CAS2)
    if save is not False:
        plt.savefig(save)
    plt.show()


best_model_loss = architecture(inpSize)
#best_model_loss.load_weights(SAVE_PATH + "//model_intermediate//" + NAME + "_" + str(epochs_loss_val.argmin()) + ".h5")

# Ne:7440-01-9, Xe:7440-63-3
pair_prediction(best_model_loss, "7440-01-9", "7440-63-3", CriticalNom, CriticalData, DipoleNom, DipoleData,
save=SAVE_PATH + "Ne_Xe_pred_bestloss.png")
# CO2:124-38-9, O2:7782-44-7
pair_prediction(best_model_loss, "124-38-9", "7782-44-7", CriticalNom, CriticalData, DipoleNom, DipoleData,
save=SAVE_PATH + "CO2_O2_pred_bestloss.png")

#best_model_RMSE = architecture(inpSize)
#best_model_RMSE.load_weights(SAVE_PATH + "//model_intermediate//" + NAME + "_" + str(epochs_RMSE_val.argmin()) + ".h5")

# Ne:7440-01-9, Xe:7440-63-3
#pair_prediction(best_model_RMSE, "7440-01-9", "7440-63-3", CriticalNom, CriticalData, DipoleNom, DipoleData,
#save=SAVE_PATH + "Ne_Xe_pred_bestRMSE.png")
# CO2:124-38-9, O2:7782-44-7
#pair_prediction(best_model_RMSE, "124-38-9", "7782-44-7", CriticalNom, CriticalData, DipoleNom, DipoleData,
#save=SAVE_PATH + "CO2_O2_pred_bestRMSE.png")
