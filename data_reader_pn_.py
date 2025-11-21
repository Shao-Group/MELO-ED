import pandas as pd
import random
from siacnn_models_gpu2 import data_reader1, data_reader3

#read data

def data_load_bd2(rate, d1, d2, path, num_test, num_train_valid):
    file_path = f'{path}data/seq-n20-ED7-add-'
    dataset_test, set_ = data_reader1(15, d1, d2, f'{path}data/heatmap_test_20mer.txt', num_test)
    print(set_)
    df_test = pd.DataFrame(dataset_test)
    if d1 == 1:
        dataset1 = data_reader3(d1-1, d1, f'{file_path}1.txt', 0, num_train_valid, -1)
        dataset2 = data_reader3(d1-1, d1, f'{file_path}2.txt', 0, num_train_valid, -1)

    else:
        dataset1 = data_reader3(d1-1, d1, f'{file_path}1.txt', 0, num_train_valid, -1)
        dataset2 = data_reader3(d1-1, d1, f'{file_path}2.txt', 0, num_train_valid, -1)

    dataset1_ = data_reader3(d2-1, d2+1, f'{file_path}1.txt', 0, num_train_valid, 1)
    dataset2_ = data_reader3(d2-1, d2+1, f'{file_path}2.txt', 0, num_train_valid, 1)

    dataset_p = dataset1+dataset2
    dataset_n = dataset1_+dataset2_
    datasets = dataset_p+dataset_n
    print('number_p:'+str(len(dataset_p))+', number_n:'+str(len(dataset_n)))
    print('boundary sampling2')

    sum_num = len(datasets)
    train_rg = int(sum_num*rate)
    random.shuffle(datasets)
    df_tr = pd.DataFrame(datasets[0:train_rg])
    df_v = pd.DataFrame(datasets[train_rg:])

    return df_tr, df_v, df_test, train_rg





def data_load_bd2_m(rate, d1, d2, path, num_test, num_train_valid):
    file_path = f'{path}data/seq-n20-ED7-add-'
    dataset_test, set_ = data_reader1(15, d1, d2, f'{path}data/heatmap_test_20mer.txt', num_test)
    print(set_)
    df_test = pd.DataFrame(dataset_test)
    if d1 == 1:
        dataset1 = data_reader3(d1-1, d1, f'{file_path}1.txt', 0, num_train_valid, -1)
        dataset2 = data_reader3(d1-1, d1, f'{file_path}2.txt', 0, num_train_valid, -1)

    else:
        dataset1 = data_reader3(d1-1, d1, f'{file_path}1.txt', 0, num_train_valid, -1)
        dataset2 = data_reader3(d1-1, d1, f'{file_path}2.txt', 0, num_train_valid, -1)

    dataset1_ = data_reader3(d2-1, d2+1, f'{file_path}1.txt', 0, num_train_valid, 1)
    dataset2_ = data_reader3(d2-1, d2+1, f'{file_path}2.txt', 0, num_train_valid, 1)

    dataset_p = dataset1+dataset2
    dataset_n = dataset1_+dataset2_
    datasets = dataset_p+dataset_n
    print('number_p:'+str(len(dataset_p))+', number_n:'+str(len(dataset_n)))
    print('boundary sampling2')

    sum_num = len(datasets)
    train_rg = int(sum_num*rate)
    random.shuffle(datasets)
    df_tr = pd.DataFrame(datasets[0:train_rg])
    df_v = pd.DataFrame(datasets[train_rg:])

    return df_tr, df_v, df_test, train_rg
