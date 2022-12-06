import os
from tkinter import *

from pyreadline import execfile


def call_GUI1():
    win1 = Toplevel(root)

    os.system('python grid.py')



    return

def call_GUI2():
    win1 = Toplevel(root)

    os.system('python random_sear.py')



    return
def call_GUI3():
    win1 = Toplevel(root)

    os.system('python main.py')



    return
def call_GUI4():
    win1 = Toplevel(root)

    os.system('python optuna_sear.py')



    return
def call_GUI5():
    win1 = Toplevel(root)

    os.system('python bayes_sear.py')



    return
def call_GUI6():
    win1 = Toplevel(root)

    os.system('python hyperas_sear.py')



    return
def call_GUI7():
    win1 = Toplevel(root)

    os.system('python wann_search.py')



    return
def call_GUI8():
    win1 = Toplevel(root)

    os.system('python neat_search.py')



    return
def call_GUI9():
    win1 = Toplevel(root)

    os.system('python halv_grid.py')



    return
def call_GUI10():
    win1 = Toplevel(root)

    os.system('python halv_randm.py')



    return
def call_GUI11():
    win1 = Toplevel(root)

    os.system('python keras_tun.py')



    return





# the first gui owns the root window
if __name__ == "__main__":
    root = Tk()
    root.title('Caller GUI')
    root.minsize(1170, 800)
    lab0= Label(root, text="Hyperparameters optimization methods",fg='black',bg='white',font=('Helvetica bold',20)).grid(row=0, column=5)
    lab00=Label(root, text="                 ").grid(row=1, column=0)
    button_1 = Button(root, text='GRID SEARCH', width='20', height='10', command=call_GUI1)
    button_1.grid(row=1, column=1)
    lab1=Label(root, text="                 ").grid(row=1, column=2)
    button_2 = Button(root, text='RANDOM SEARCH', width='20', height='10', command=call_GUI2)
    button_2.grid(row=1, column=3)
    lab2 = Label(root, text="                 ").grid(row=1, column=4)
    button_3 = Button(root, text='Manual Search', width='20', height='10', command=call_GUI3)
    button_3.grid(row=1, column=5)
    lab3 = Label(root, text="                 ").grid(row=1, column=6)
    button_4 = Button(root, text='Optuna', width='20', height='10', command=call_GUI4)
    button_4.grid(row=1, column=7)
    lab4 = Label(root, text="                 ").grid(row=1, column=8)
    button_5 = Button(root, text='Bayesian', width='20', height='10', command=call_GUI5)
    button_5.grid(row=1, column=9)
    labn = Label(root, text="                 ").grid(row=2, column=8)
    labnn = Label(root, text="                 ").grid(row=3, column=8)
    labn = Label(root, text="                 ").grid(row=4, column=8)
    labnn = Label(root, text="                 ").grid(row=5, column=8)
    labn = Label(root, text="                 ").grid(row=6, column=8)
    labnn = Label(root, text="                 ").grid(row=7, column=8)
    button_6 = Button(root, text='Hyperas (dev on progress)', width='20', height='10', command=call_GUI6)
    button_6.grid(row=8, column=1)
    lab5= Label(root, text="").grid(row=6, column=1)
    button_7 = Button(root, text='Neuro-Evol WANN', width='20', height='10', command=call_GUI7)
    button_7.grid(row=8, column=3)
    button_8 = Button(root, text='Neuro-Evol NEAT', width='20', height='10', command=call_GUI8)
    button_8.grid(row=8, column=5)
    button_9 = Button(root, text='Halving Grid', width='20', height='10', command=call_GUI9)
    button_9.grid(row=8, column=7)
    button_10 = Button(root, text='Halving Random', width='20', height='10', command=call_GUI10)
    button_10.grid(row=8, column=9)

    labn = Label(root, text="                 ").grid(row=9, column=8)
    labnn = Label(root, text="                 ").grid(row=10, column=8)
    labn = Label(root, text="                 ").grid(row=11, column=8)
    labnn = Label(root, text="                 ").grid(row=12, column=8)
    labn = Label(root, text="                 ").grid(row=13, column=8)
    labnn = Label(root, text="                 ").grid(row=14, column=8)
    button_11 = Button(root, text='Keras Tuner', width='20', height='10', command=call_GUI11)
    button_11.grid(row=15, column=5)
    root.mainloop()