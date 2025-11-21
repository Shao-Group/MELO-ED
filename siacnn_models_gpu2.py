import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

#######################################DATA READER######################################################

def data_reader1(eds, d1, d2, id_, i_rang):
    with open(id_, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n').split(' ')
    ed = [i+1 for i in range(eds)]
    set_ = {}
    Po = []
    Ne = []
    #f = open(path+'heatmap_test_'+str(len(lines[0][0]))+'mer.txt', 'w')
    for i in range(len(lines)):
        if int(lines[i][2]) in ed:
            if int(lines[i][2]) not in set_.keys():
                set_[int(lines[i][2])] = 1
                #print(lines[i][0], lines[i][1], lines[i][2], file=f)
            elif (set_[int(lines[i][2])] >= i_rang):continue
            else:
                set_[int(lines[i][2])] += 1
                #print(lines[i][0], lines[i][1], lines[i][2], file=f)
                if int(lines[i][2])<=d1:
                    Po.append([lines[i][0], lines[i][1], lines[i][2], -1])
                elif int(lines[i][2])>=d2:
                    Ne.append([lines[i][0], lines[i][1], lines[i][2], 1])
    dataset = Po+Ne
    random.shuffle(dataset)
    return dataset, set_


def data_reader3(eds, ede, id_, i_ranga, i_rangb, label):
    with open(id_, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n').split(' ')
    eds, ede = [i+1 for i in range(eds)], [i+1 for i in range(ede)]
    ed = list(set(ede).difference(set(eds)))
    set_all = []
    for j in range(len(ed)):
        set_ = []
        for i in range(len(lines)):
            if int(lines[i][2]) == ed[j]:
                set_.append([lines[i][0], lines[i][1], lines[i][2], label])

        set_all += set_[i_ranga:i_rangb]

    return set_all
    

def leng_fea(seq):
    leng = []
    A = torch.zeros(4)
    C = torch.zeros(4)
    G = torch.zeros(4)
    T = torch.zeros(4)
    A[0] += 1
    C[1] += 1
    G[2] += 1
    T[3] += 1
    fea = [A, C, G, T]
    word = ['A', 'C', 'G', 'T']

    for i in range(len(seq)):
        for j in range(len(word)):
            if seq[i] == word[j]:
                leng.append(fea[j])

    leng = torch.stack(leng)

    return leng


def aby_sep(df):

    labels_ = list(df[3])
    labels = []
    for i in range(len(labels_)):
        if labels_[i] == 1:
            labels.append(0)
        else:
            labels.append(1)
    #labels = torch.tensor(labels)

    samples_a = df[0]
    samples_b = df[1]
    seq_a = []
    for i in range(len(samples_a)):
        seq_a.append(leng_fea(samples_a[i]))

    seq_b = []
    for i in range(len(samples_b)):
        seq_b.append(leng_fea(samples_b[i]))

    return seq_a, seq_b, torch.tensor(labels_), torch.tensor(labels) 

def ed_sp(df):
    ed_dict = {}
    for i in range(len(df[1])):
        if int(df[2][i]) not in ed_dict.keys():
            ed_dict[int(df[2][i])] = [[df[0][i], df[1][i]]]
        else:
            ed_dict[int(df[2][i])].append([df[0][i], df[1][i]])

    return ed_dict

#################################################Batch set###############################################################
def mini_batch_cnn1(x, i, batch_size):
    x_range = x[i*batch_size : (i+1)*batch_size]
    batch_x = torch.cat([torch.unsqueeze(i.t(), 0) for i in x_range], 0)
    batch_x = torch.unsqueeze(batch_x, 1)
    return batch_x

#########################################Inp_module######################################################################

class Inp_Layer1(nn.Module):
    def __init__(self, in_ch, out_ch, in_dim, out_max):#out_size,
        super(Inp_Layer1, self).__init__()

        self.cnn1 = nn.Conv2d(in_ch, out_ch, (in_dim, 2), stride=1, padding="same")
        self.cnn2 = nn.Conv2d(in_ch, out_ch, (in_dim, 3), stride=1, padding="same")
        self.cnn3 = nn.Conv2d(in_ch, out_ch, (in_dim, 4), stride=1, padding="same")
        self.cnn4 = nn.Conv2d(in_ch, out_ch, (in_dim, 5), stride=1, padding="same")
        self.cnn5 = nn.Conv2d(in_ch, out_ch, (in_dim, 6), stride=1, padding="same")
        self.maxpool = nn.MaxPool2d((1, out_max), stride=(1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):

        out1 = F.relu(self.cnn1(x))
        out1 = self.maxpool(out1)
        out2 = F.relu(self.cnn2(x))
        out2 = self.maxpool(out2)
        out3 = F.relu(self.cnn3(x))
        out3 = self.maxpool(out3)
        out4 = F.relu(self.cnn4(x))
        out4 = self.maxpool(out4)

        out5 = F.relu(self.cnn5(x))
        out5 = self.maxpool(out5)
        out = F.normalize(torch.cat((out1, out2, out3, out4, out5), 1))

        return out

class Inp_Model_2(nn.Module):
    def __init__(self):#out_size,
        super(Inp_Model_2, self).__init__()

        self.inp_layer1 = Inp_Layer1(1, 10, 4, 3)
        self.inp_layer2 = Inp_Layer1(50, 10, 4, 3)

        self.flat = nn.Flatten()

    def forward(self, x):

        out = self.inp_layer1(x)
        out = self.inp_layer2(out)
        out = self.flat(out)

        return out




