import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from siacnn_models_gpu2 import *
from new_loss_model import *
from data_reader_pn_ import *


def Training_Evaluation_Parameter_Set(d1, d2, rate, a_, path, num_test, num_train_valid, batch_size, delta, m_dim, num_b):

    data_path = f'{path}'

    models_path = f'{path}models/'
    os.makedirs(models_path, exist_ok=True)

    results_path = f'{path}results/'
    os.makedirs(results_path, exist_ok=True)


    df_tr, df_v, df_test, train_rg = data_load_bd2(rate, d1, d2, data_path, num_test, num_train_valid)

    train_a, train_b, train_t, train_y = aby_sep(df_tr)
    valid_a, valid_b, valid_t, valid_y = aby_sep(df_v)
    test_a, test_b, test_t, test_y = aby_sep(df_test)

    eds = ed_sp(df_tr)
    print('edits number (train)')
    ed_num_train = []
    for i in sorted(eds.keys()):
        print('ed = ', i, ': ', len(eds[i]))
        ed_num_train.append([i, len(eds[i])])

    eds_t = ed_sp(df_test)
    print('data loaded')
    #for num_b in num_b_set:

    cnnk = Inp_Model_2().to(device)
    flat_dim = cnnk(a_).shape[1]
    out_dim = num_b*m_dim
    siacnn2 = SiamNNL1_nm(cnnk, flat_dim, out_dim).to(device)
    ID = f'{num_b}k_{m_dim}m_({d1}-{d2})s_{delta}delta'
    print(f'{ID} model construct')
    trainer1 = Trainer1(train_a, train_b, train_t, siacnn2, loss0, delta, batch_size)
    print('##########train start###########')
    lr = 0.002 #learning rate, initial = 0.001 
    num_epo = 30 #numbers of epoch
    loss_t = []       
    loss_v = [] 
    for i in range(3):
        lr *= 0.5
        loss1_, loss11_ = trainer1.run(num_epo, lr, valid_a, valid_b, valid_t, m_dim, num_b, device)
        loss_t += loss1_
        loss_v += loss11_ 

        torch.save(cnnk, f'{models_path}{cnnk.__class__.__name__}_{ID}.pt')
        torch.save(siacnn2, f'{models_path}{siacnn2.__class__.__name__}_{cnnk.__class__.__name__}_{ID}.pt')
        
    print('##########train end###########')
    f = h5py.File(f'{results_path}loss_acc_{cnnk.__class__.__name__}_{ID}.hdf5', 'a')
    f.create_dataset('loss_t', data=np.array(loss_t))
    f.create_dataset('loss_v', data=np.array(loss_v))

    print('###########TESTING###############')

    print('breakdown acc')
    print('train: ')
    res, acc_tr = breakdown_acc(eds, d1, d2, siacnn2, acc_count_0, batch_size, delta, m_dim, num_b, device)
    print('acc_train: '+str(acc_tr))
    print('test')
    res_t, acc_t = breakdown_acc(eds_t, d1, d2, siacnn2, acc_count_0, batch_size, delta, m_dim, num_b, device)
    print('acc_test: '+str(acc_t))

    f.create_dataset('accs', data=np.array([acc_tr, acc_t]))
    bd_res = []
    for i in sorted(res.keys()):
        bd_res.append([i, res[i]])

    bd_res_t = []
    for i in sorted(res_t.keys()):
        bd_res_t.append([i, res_t[i]])

    f.create_dataset('bd_train', data=np.array(np.array(bd_res)))
    f.create_dataset('bd_test', data=np.array(np.array(bd_res_t)))
    f.close()


#CUDA
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')


def main():
    # run example 
    N_len = 20  # sequence length N
    d1, d2 = [1, 3]  # [[1, 2], [2, 3], [2, 4], [3, 4], [3, 5]]
    m_dim = 40  # dimension of embedding vectors m
    batch_size = 100
    num_b = 20  # number of embedding vectors K
    delta = 10
    a_ = torch.rand(100, 1, 4, N_len).to(device)
    rate = 0.9  # train/valid = 9:1

    path = '/storage/home/xvy5180/work/seqhash/seq_20n/'
    # Make sure folder exists
    os.makedirs(path, exist_ok=True)

    num_test, num_train_valid = [20000, 100000]  
    # e.g., 20000 pairs for test, 100000 for train+valid

    Training_Evaluation_Parameter_Set(d1, d2, rate, a_, path, num_test, num_train_valid, batch_size, delta, m_dim, num_b)

if __name__ == "__main__":
    main()





