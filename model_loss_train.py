import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py
import math

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from functions import *

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


class SiamNNL1(nn.Module):
    def __init__(self, cnn, flat_dim, out_dim):
        super(SiamNNL1, self).__init__()
        self.cnn = cnn
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.BatchNorm1d(out_dim)
            #nn.ReLU()
            #nn.Linear(hidden_dim, out_dim),
            #nn.Sigmoid()
            #nn.Tanh()
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = self.fc1(out)

        return out

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2


def loss0(a, b, t, x_0, delta, m_dim, num_b, tp):

    hash_a = a.reshape([a.shape[0], num_b, m_dim])
    hash_b = b.reshape([b.shape[0], num_b, m_dim])
    ed = torch.norm(hash_a - hash_b, p=2, dim=2)
    d = torch.min(ed, 1).values

    loss = torch.max(x_0, -(d-delta)*t)

    if tp == 'sum':
        loss_ = torch.sum(loss)
        return loss_
    elif tp == 'mean':
        loss_ = torch.mean(loss)
        return loss_


class Trainer1:
    def __init__(self, x_a, x_b, t, model, loss_f, delta, batchsize):
        super(Trainer1, self).__init__()
        self.x_a = x_a
        self.x_b = x_b
        self.t = t
        self.model = model
        self.loss_f = loss_f
        self.delta = delta
        self.batch_size = batchsize

    def run(self, epo, lr, v_a, v_b, v_t, m_dim, num_b, device):#, state1

        loss_ = []
        loss1_ = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        for epoch in range(epo):
            optimizer.zero_grad()
            #D_t = 0
            final_loss_t = 0
            for i in range(int(len(self.x_a)/self.batch_size)):
                mini_inputa = mini_batch_cnn1(self.x_a, i, self.batch_size).to(device)
                mini_inputb = mini_batch_cnn1(self.x_b, i, self.batch_size).to(device)
                batch_t = self.t[i*self.batch_size : (i+1)*self.batch_size].to(device)

                out1, out2 = self.model(mini_inputa, mini_inputb)
                loss = self.loss_f(out1, out2, batch_t, torch.zeros(self.batch_size).to(device), self.delta, m_dim, num_b, 'sum')
                loss.backward()
                optimizer.step()
                final_loss_t += float(loss)

            final_loss_v = 0
            for i in range(int(len(v_a)/self.batch_size)):
                mini_inputa = mini_batch_cnn1(v_a, i, self.batch_size).to(device)
                mini_inputb = mini_batch_cnn1(v_b, i, self.batch_size).to(device)
                batch_t = v_t[i*self.batch_size : (i+1)*self.batch_size].to(device)

                out1, out2 = self.model(mini_inputa, mini_inputb)
                loss1 = self.loss_f(out1, out2, batch_t, torch.zeros(self.batch_size).to(device), self.delta, m_dim, num_b, 'sum')
                final_loss_v += float(loss1)

            loss_.append(final_loss_t/int(len(self.x_a)/self.batch_size))
            loss1_.append(final_loss_v/int(len(v_a)/self.batch_size))
            #if epoch % 10 == 0:
            print('epoch:', epoch, 'loss_t:', float(final_loss_t/int(len(self.x_a)/self.batch_size)), 'loss_v:', float(final_loss_v/int(len(v_a)/self.batch_size)))

        return loss_, loss1_

def round_0(x, th, device):
    x_n = torch.where(x > th, torch.tensor(0).to(device), torch.tensor(1).to(device))
    return x_n

def acc_count_0(a, b, delta, m_dim, num_b, device):

    hash_a = a.reshape([a.shape[0], num_b, m_dim])
    hash_b = b.reshape([b.shape[0], num_b, m_dim])
    ed = torch.norm(hash_a - hash_b, p=2, dim=2)
    d = torch.min(ed, 1).values
    y_ = round_0(d, delta, device)
    #y_ = round_0(d/math.sqrt(m_dim), delta, device)

    return y_ #ed, d, y_

def acc_count_nm(a, b, delta, m_dim, num_b, device):

    hash_a = a.reshape([a.shape[0], num_b, m_dim])
    hash_b = b.reshape([b.shape[0], num_b, m_dim])
    ed = torch.norm(hash_a - hash_b, p=2, dim=2)
    d = torch.min(ed, 1).values
    y_ = round_0(d/math.sqrt(m_dim), delta, device)

    return y_ #ed, d

def acc_count_s(a, b, delta, m_dim, num_b, device):

    hash_a = a.reshape([a.shape[0], num_b, m_dim])
    hash_b = b.reshape([b.shape[0], num_b, m_dim])
    ed = torch.norm(hash_a - hash_b, p=2, dim=2)
    d = F.sigmoid(torch.min(ed, 1).values)
    y_ = round_0(d, delta, device)

    return y_ #ed, d



def acc_test_batch(test_a, test_b, test_y, delta, batchsize, siacnn, acc_count_0, m_dim, num_b, device):
    scores = []
    for i in range(int(len(test_a)/batchsize)):
        minit_inputa1 = mini_batch_cnn1(test_a, i, batchsize)
        minit_inputb1 = mini_batch_cnn1(test_b, i, batchsize)
        minit_y = test_y[i*batchsize : (i+1)*batchsize]

        out1_, out2_ = siacnn(minit_inputa1.to(device), minit_inputb1.to(device))
        score_t = acc_count_0(out1_, out2_, delta, m_dim, num_b, device)
        scores += score_t
    num1 = int(len(test_a)/batchsize)*batchsize
    num2 = len(test_a) - num1
    if num2 != 0:
        out1_, out2_ = siacnn(mini_batch_cnn1(test_a[num1:], 0, num2).to(device), mini_batch_cnn1(test_b[num1:], 0, num2).to(device))
        score_t = acc_count_0(out1_, out2_, delta, m_dim, num_b, device)
        #score, _ = acc_fun0(out1_, out2_, test_y[num1:].to(device), th_x, m_dim, num_b, device)
        scores += score_t
    equal_to_T = (torch.eq(torch.tensor(scores), test_y) == True)
    count = torch.sum(equal_to_T).tolist()

    return count/int(len(scores)), count

def breakdown_acc(eds, d1, d2, siacnn, acc_count_0, batchsize, delta, m_dim, num_b, device):
    res = {} 
    resl = 0
    c_num = 0
    for key in eds.keys():
        seq_a = []
        seq_b = []
        for j in range(len(eds[key])):
            seq_a.append(leng_fea(eds[key][j][0]))
            seq_b.append(leng_fea(eds[key][j][1]))
        if key<=d1:
            test_y = torch.ones(len(eds[key]))
        elif key>=d2:
            test_y = torch.zeros(len(eds[key]))
        acc, count = acc_test_batch(seq_a, seq_b, test_y, delta, batchsize, siacnn, acc_count_0, m_dim, num_b, device)
        res[key] = acc
        c_num += len(seq_a)
        resl += count
    
    print ('acc_all: ', resl/c_num)
    for i in sorted(res.keys()):
        print('ed = ', i, ' acc = ', res[i])

    return res, resl/c_num

