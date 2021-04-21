# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:26:12 2021

@author: chaya
"""
#importing library
import pandas as pd
import numpy as np
# open source Deep Leaning library
import tensorflow as tf 
from tensorflow import keras
# For plotting figures
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=15)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


#load mnist data
# Load MNIST Data
kannada_test = pd.read_csv("C:/Users/chay/Desktop/mini/datasets/test.csv")
kannada_train = pd.read_csv("C:/Users/chaya/Desktop/mini/datasets/train.csv")

