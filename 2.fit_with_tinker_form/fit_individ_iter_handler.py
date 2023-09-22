import sys

box_prev =  sys.argv[2]
box = sys.argv[1]
print('Curr: %-10s, Prev: %-10s'%(box, box_prev))
import numpy as np
from tqdm import tqdm, trange
import os
import numpy as np
import pickle


a_list_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/alist.pkl'
a_list_f = open(a_list_fname, 'rb')
a_list = pickle.load(a_list_f) 
a_list_f.close()
print('alist', a_list)

import subprocess
from datetime import date



line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, 1.0, box_prev, 1.0)
print(line)
result = subprocess.run(line, shell=True, check=True)
for i in trange(1, len(a_list)):
    line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, a_list[i], box, a_list[i-1])
    print(line)
    result = subprocess.run(line, shell=True, check=True)
