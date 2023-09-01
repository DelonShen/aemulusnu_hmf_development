from aemulusnu_massfunction.utils import *
from aemulusnu_massfunction.massfunction import *

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle

cosmos_f = open('../data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()

X = []
Y = []
idx = {}
cidx = 0
weird_boxes = ['Box63_1400', 'Box35_1400', 'Box_n50_38_1400', 'Box5_1400']
for box in cosmo_params:
    if(box in weird_boxes):
        continue
    X += [box]
    idx[box] = cidx
    cidx+=1
    Y += [[cosmo_params[box][key] for key in cosmo_params[box]]]
    
X = np.array(X)
Y = np.array(Y)

from aemulusnu_massfunction.utils import Standardizer

# Assuming Y is your input matrix with shape (n_samples, n_features)
# Create an instance of StandardScaler
scaler = Standardizer()

# Fit the scaler to the data
scaler.fit(Y)

# Standardize the data
Y = scaler.transform(Y)

print(len(X))

def find_most_central_datapoint_index(Y):
    centroid = np.mean(Y, axis=0)
    distances = np.linalg.norm(Y - centroid, axis=1)
    most_central_index = np.argmin(distances)
    return most_central_index
print('most central: ', X[find_most_central_datapoint_index(Y)])

def dist(a,b):
    return np.sqrt(np.sum(np.square(np.array(Y[b])-np.array(Y[a]))))

edges = [[dist(i,j) for i in range(len(X))] for j in range(len(X))]


import sys

def find_min_key(key, mst_set, n):
    min_key = sys.maxsize
    min_index = -1
    for i in range(n):
        if key[i] < min_key and not mst_set[i]:
            min_key = key[i]
            min_index = i
    return min_index

def prim_mst(edges, n):
    parent = [None] * n
    key = [sys.maxsize] * n
    mst_set = [False] * n

    key[0] = 0  # Start with the first node as the root
    parent[0] = -1  # Root has no parent

    for _ in range(n - 1):
        u = find_min_key(key, mst_set, n)
        mst_set[u] = True

        for v in range(n):
            if edges[u][v] > 0 and not mst_set[v] and edges[u][v] < key[v]:
                key[v] = edges[u][v]
                parent[v] = u

    mst = [[0] * n for _ in range(n)]
    for i in range(1, n):
        mst[i][parent[i]] = edges[i][parent[i]]
        mst[parent[i]][i] = edges[parent[i]][i]

    return mst

n = len(edges)

mst = prim_mst(edges, n)

from collections import deque

def bfs_traversal(mst, start):
    tot = 0
    n = len(mst)
    visited = [False] * n
    parent = [None] * n

    queue = deque()
    queue.append(start)
    visited[start] = True

    while queue:
        node = queue.popleft()
        print("%-10s"%(X[node]), end='\t ')
        if parent[node] is not None:
            prev = parent[node]
            dist = np.sqrt(np.sum(np.square(np.array(Y[idx[X[node]]])- np.array(Y[idx[X[prev]]]))))
            tot += dist

            print("Prev: %-10s\t distance:%.2f"%(X[parent[node]], dist),  end='')
        print()

        for neighbor in range(n):
            if mst[node][neighbor] > 0 and not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = node
    assert(all(visited))
    print(tot)

start_node = idx['Box_n50_0_1400'] # Starting node for traversal

bfs_traversal(mst, start_node)


from collections import deque

import subprocess
import time

def run_fit_iter(box, prev_box):
    #run the fit job for this guy 
    subprocess.run('./fit_individ_handler.sh %s %s'%(box, prev_box), shell=True)

def bfs_traversal_run_jobs(mst, start):
    n = len(mst)
    visited = [False] * n
    parent = [None] * n

    queue = deque()
    queue.append(start)
    visited[start] = True

    while queue:
        node = queue.popleft()
        #check if parent still running
        curr_parent = parent[node]
        if(curr_parent is not None):
            squeue_output = subprocess.check_output(['squeue', '-u', 'delon', '-n', 'fit-iter-handler-%s'%(X[curr_parent]), '-h']).decode().strip()
#             print('sq', squeue_output)
            if not squeue_output: #if job done running 
                print('Running fit for %s'%(X[node]))
                run_fit_iter(X[node], X[curr_parent])
                #add its neighboros to the queue
                for neighbor in range(n):
                    if mst[node][neighbor] > 0 and not visited[neighbor]:
                        queue.append(neighbor)
                        visited[neighbor] = True
                        parent[neighbor] = node
            else: #parent job not done running
#                 print('Fit for %s still running'%(X[curr_parent]))
                queue.append(node)
        else:
            for neighbor in range(n):
                if mst[node][neighbor] > 0 and not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
                    parent[neighbor] = node
#         print('-----END----')

    assert(all(visited))

start_node = idx['Box_n50_0_1400'] # Starting node for traversal

bfs_traversal_run_jobs(mst, start_node)
print('Iterataive Graph Fitting Traversal Done!')
