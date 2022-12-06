## LIBRARIES IMPORT


#from tensorflow.python.keras.optimizers import RMSprop
""" gp.py
Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp

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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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
NAME = "test_grid_search"
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


def architecture( nb_laters ,nb_neurons):
    initializer = "glorot_uniform"
    model = tf.keras.Sequential()

    for i in range(nb_laters):
        model.add(
            tf.keras.layers.Dense(nb_neurons, input_dim=11, kernel_initializer='uniform', activation='softsign'))


    model.add(tf.keras.layers.Dense(1, kernel_initializer='uniform'))
    model.compile(loss="mean_squared_error", optimizer= tf.optimizers.RMSprop(0.001, rho=0.9, epsilon=None, decay=0.0), metrics=[ tf.keras.losses.MSE,"mean_absolute_percentage_error"])
    return model


def fitness(params):
    model = architecture(nb_laters=np.int(params[0]),
                         nb_neurons=np.int(params[1])

                         )

    # named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_train,
                         y=Y_train,
                         epochs=epoch(),
                         batch_size=Batch(),
                         validation_split=acti())

    # return the validation accuracy for the last epoch.
    accuracy = blackbox.history['mean_squared_error'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print(blackbox.history['mean_squared_error'][-1])
    print()
    # print the parameters
    print(params)
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.grid(row=9, column=1)

    mylist = Listbox(root, width=140, yscrollcommand=scrollbar.set)
    n=epoch()
    for i in range(0,n):
        accuracy = blackbox.history['mean_squared_error'][i]
        mylist.insert(END, "Accuracy: {0:.2%}".format(accuracy))

    mylist.grid(row=9, column=1, rowspan=4,
                columnspan=6)
    scrollbar.config(command=mylist.yview)
    del model

    return accuracy
def bayes():

 bounds = np.array([[layers(), opti()],[init_mod(),neurons()]])



 xp, yp = bayesian_optimisation(n_iters=iteration(), sample_loss=fitness,
                               bounds=bounds,
                               n_pre_samples=splits())
 xp_hat = np.round(xp[np.array(yp).argmin(), :])


 print("best hyperparams values are: ")
 print()
 print(np.round(xp_hat))
 res=np.round(xp_hat)
 lab00 = Label(root, text="best hyperparams values are: ").grid(
     row=13, column=0)
 lab0 = Label(root, text=res).grid(
     row=13, column=1)
## HYPERPARAMETERS
print("\n----------------------------------------")
print("Hyperparameters...")

# BATCH SIZE
BATCH_SIZE = [128]


# OPTIMIZER
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#lr = [0.001,0.01,0.05,0.09,0.1,0.2]
weight_constraint = [1, 2, 3, 4]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
OPTIMIZER = tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)  # todo: optimize optimizer
# OPTIMIZER = tf.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
# OPTIMIZER = tf.optimizers.Adam(lr=1e-3)
# OPTIMIZER = tf.optimizers.SGD(lr=0.01, clipnorm=1.)
# OPTIMIZER = tf.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# LOSS
#LOSS = tf.keras.losses.MSE
LOSS = "mean_absolute_percentage_error"

from tkinter import *
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
    layer=num
    lab11 = Label(root, text=layer).grid(row=0, column=3)
    e1.insert(0,'')
    return layer

def neurons():
    v = e6.get()

    num = int(v)
    print(num)
    neuron=num
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
    op=int(opt)
    optimi=op
    print(optimi)
    lab21 = Label(root, text=optimi).grid(row=1, column=3)
    e2.insert(0,"")
    return optimi
def acti():

    act = e8.get()
    print(act)
    vald=float(act)
    activi=vald
    print(activi)
    lab81 = Label(root, text=activi).grid(row=7, column=3)
    e8.insert(0,"")
    return activi
def init_mod():

    initia = e5.get()
    print(initia)
    init=int(initia)
    ini=init
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

if __name__ == "__main__":
    root = Tk()
    root.title('Bayesian Search')
    root.minsize(1100, 800)
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.grid(row=9, column=1)

    mylist = Listbox(root, width=140, yscrollcommand=scrollbar.set)
    mylist.grid(row=9, column=1, rowspan=4,
                columnspan=6)
    scrollbar.config(command=mylist.yview)

    button_1 = Button(root, text='Start Bayesian Search', width='20', height='5', command=bayes)
    button_1.grid(row=13, column=4)
    lab1= Label(root, text="Layers (low)").grid(row=0)
    lab2= Label(root, text="Layers (high)").grid(row=1)
    lab3= Label(root, text="Epoch").grid(row=2)
    lab4= Label(root, text="Batch size").grid(row=3)
    lab5 = Label(root, text="Neurons (low)").grid(row=4)
    lab6 = Label(root, text="Neurons (high)").grid(row=5)
    lab7 = Label(root, text="Num of pre samples").grid(row=6)
    lab8 = Label(root, text="Validation split").grid(row=7)
    lab9 = Label(root, text="Iteration number").grid(row=8)
    e9=Entry(root)
    e9.grid(row=8,column=1)
    e9.insert(0,"20")
    e8=Entry(root)
    e8.grid(row=7,column=1)
    e8.insert(0,"0.15")
    e7= Entry(root)
    e7.grid(row=6, column=1)
    e6 = Entry(root)
    e6.grid(row=5, column=1)
    e6.insert(0,"150")
    e5= Entry(root)
    e5.grid(row=4, column=1)
    e5.insert(0,"10")
    e4= Entry(root)
    e4.grid(row=3, column=1)
    e4.insert(0,"16")
    e3= Entry(root)
    e3.grid(row=2, column=1)
    e3.insert(0,"100")
    e1 = Entry(root)
    e1.insert(0, "2")
    e2 = Entry(root)
    e2.insert(0, "6")
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e7.insert(0,"5")
    button_3 = Button(root, text='add layer (high) ', width='20', height='2', command=opti)
    button_3.grid(row=1, column=2)


    button_2 = Button(root, text='add layer (low) ', width='20', height='2', command=layers)
    button_2.grid(row=0, column=2)


    button_4=Button(root, text='add Epoch number ', width='20', height='2', command=epoch)
    button_4.grid(row=2, column=2)

    button_5 = Button(root, text='add Batch size ', width='20', height='2', command=Batch)
    button_5.grid(row=3, column=2)

    button_6 = Button(root, text='add neuron (low) ', width='20', height='2', command=init_mod)
    button_6.grid(row=4, column=2)

    button_7 = Button(root, text='add neuron (high) ', width='20', height='2', command=neurons)
    button_7.grid(row=5, column=2)

    button_8 = Button(root, text='add num pre samples ', width='20', height='2', command=splits)
    button_8.grid(row=6, column=2)

    button_9 = Button(root, text='add validation split ', width='20', height='2', command=acti)
    button_9.grid(row=7, column=2)

    button_10 = Button(root, text='add iteration number ', width='20', height='2', command=iteration)
    button_10.grid(row=8, column=2)

    layer=[2]
    optimi=[6]
    epo=1
    batch=[1]
    ini=[10]
    neuron=[150]
    split=5
    activi=0.15
    ite=20

    lab11= Label(root,text=layer).grid(row=0,column=3)
    lab21= Label(root,text=optimi).grid(row=1,column=3)
    lab31 = Label(root, text=epo).grid(row=2, column=3)
    lab41 = Label(root, text=batch).grid(row=3, column=3)
    lab51 = Label(root, text=ini).grid(row=4, column=3)
    lab61 = Label(root, text=split).grid(row=6, column=3)
    lab71 = Label(root, text=neuron).grid(row=5, column=3)
    lab81 = Label(root, text=activi).grid(row=7, column=3)
    lab91 = Label(root, text=ite).grid(row=8, column=3)


    root.mainloop()
