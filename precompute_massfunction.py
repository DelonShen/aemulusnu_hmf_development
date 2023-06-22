import sys 
import numpy as np

from utils import *
from massfunction import *

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import dill as pickle
from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline


box = sys.argv[1]

cosmos_f = open('data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()

a_list_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/alist.pkl'
a_list_f = open(a_list_fname, 'rb')
a_list = pickle.load(a_list_f) 
a_list_f.close()


cosmo = cosmo_params[box]
mass_function = MassFunction(cosmo) 

for a in tqdm(a_list):
    mass_function.compute_dlnsinvdM(a)




with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_massfunction.pkl"%(box), "wb") as f:
    pickle.dump([mass_function.dlnσinvdMs, mass_function.Pka], f)
