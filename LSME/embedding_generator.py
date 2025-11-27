import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py

from torch.utils.data import DataLoader, Dataset
import numpy as np
from functions import *
from model_loss_train import *


def embedding_generate(d1, d2, num_b, m_dim, delta, models_path, embedding_path, seq_a, seq_b, batch_size):

    ID = f'{num_b}k_{m_dim}m_({d1}-{d2})s_{delta}delta'
    incp_name = f'{models_path}Inp_Model_2_{ID}.pt'
    siacnn_name = f'{models_path}SiamNNL1_nm_Inp_Model_2_{ID}.pt'
    if not os.path.exists(siacnn_name):
        print(f"{siacnn_name} not found, continue...")
    else:
        incp = torch.load(incp_name, weights_only=False)
        siacnn = torch.load(siacnn_name, weights_only=False)
        print(ID+' model loaded')

    for i in range(int(len(seq_a)/batch_size)):
        mini_inputa = mini_batch_cnn1(seq_a, i, int(batch_size)).to(device)
        mini_inputb = mini_batch_cnn1(seq_b, i, int(batch_size)).to(device)
        out1, out2 = siacnn(mini_inputa, mini_inputb)
        out1 = out1.reshape([out1.shape[0], num_b, m_dim]).detach().cpu().resolve_conj().resolve_neg().numpy()
        out2 = out2.reshape([out2.shape[0], num_b, m_dim]).detach().cpu().resolve_conj().resolve_neg().numpy()
        if i == 0:
            embd_a = out1
            embd_b = out2
        else:
            embd_a = np.concatenate((embd_a, out1), axis=0)
            embd_b = np.concatenate((embd_b, out2), axis=0)

    f = h5py.File(f'{embedding_path}embedding_{ID}.hdf5', 'w')
    f.create_dataset('embed_a', data=np.array(embd_a))
    f.create_dataset('embed_b', data=np.array(embd_b))
    print('embedding created')
    f.close()

#load CUDA
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

print('##################################DATA CODING####################################')

data_path = '~/data'

f = open(f'{data_path}heatmap_test_20mer.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip('\n').split(' ')

seq_a = []
seq_b = []
for i in range(10000):
#for i in range(len(lines)):
    seq_a.append(leng_fea(lines[i][0]))
    seq_b.append(leng_fea(lines[i][1]))

print('#################################embedding generating#############################')

def main():
    d1, d2 = [1, 3]
    m_dim = 40
    batch_size = 1000
    delta = 10
    num_b = 20

    path = '~/'
    models_path = f'{path}models/'

    embedding_path = f'{path}embedding/'
    os.makedirs(embedding_path, exist_ok=True)
    embedding_generate(d1, d2, num_b, m_dim, delta, models_path, embedding_path, seq_a, seq_b, batch_size)

if __name__ == "__main__":
    main()



