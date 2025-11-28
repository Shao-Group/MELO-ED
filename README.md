Introduction
==============

This is the code of work [MELO-ED](https://www.biorxiv.org/content/10.1101/2025.11.23.689944v1), including a training of a deep Locality Sensitive Multi-Embedding (LSME) function [LSME](https://github.com/Shao-Group/MELO-ED/tree/main/LSME) and a neighbor search procedure [neighbor_index_experiments](https://github.com/Shao-Group/MELO-ED/tree/master/neighbor_index_experiments). 

Usage
==============

There is an example of using our code: a framework to train an (d1, d2)-LSME function with customized parameters: N, length of sequence; k, number of embedding; m dimensions, and through δ. You can train the model using the simulation dataset provided at [zenodo](https://zenodo.org/uploads/17728758), with N = 20 or N = 100, or generate simulation sequences of the length you need with the code at [simulation](https://github.com/Shao-Group/lsb-learn/tree/master/simulation).

- Installation: `python vision >= 3.9`; `conda --version 24.11.3`;  `torch (2.7.1+cu128)`

- Train/valid/test data loading by `data_reader_pn_.py`: 

``` python
from data_reader_pn_ import data_load_bd2 

df_tr, df_v, df_test = data_load_bd2(rate, d1, d2, data_path, num_test, num_train_valid)
```
> which require the following parameters: `(d1, d2)`; `rate` of train and valid (like `rate = 0.9`means 90% of whole set for training and 10% for valdiation); `num_test` is the number you need for each edit distance for testing; `num_train_valid` is the number you need for each edit distance for training and validation.

- Model training: the structure of Siamese model, hinge loss, and train functions are saved in `model_loss_train.py`. Run `python runner.py` to train the model and evaluate its performance by displaying the loss per episode and accuracy. A quick example show how to train a (1, 3)-LSME function for sequence length N = 20, k = 20, m = 40, δ = 10:

``` python
def main_20n():
    # run example 
    N_len = 20  # sequence length N
    d1, d2 = [1, 3] 
    m_dim = 40  # dimension of embedding vectors m
    batch_size = 10000
    num_b = 20  # number of embedding vectors K

    delta = 10 # δ
    a_ = torch.rand(100, 1, 4, N_len).to(device)
    rate = 0.9  # train/valid = 9:1

    path = '~/'
    data_path = '~/data/'

    num_test, num_train_valid = [20000, 100000]
    # e.g., 20000 pairs for test, 100000 for train+valid

    df_tr, df_v, df_test = data_load_bd_20m(rate, d1, d2, data_path, num_test, num_train_valid)
    Training_Evaluation_Parameter_Set(d1, d2, a_, path, df_tr, df_v, df_test, batch_size, delta, m_dim, num_b)

```

> The structure and parameter of trained model will be saved at `~/models/SiamNNL1_nm_Inp_Model_2_20k_40m_(1-3)s_10delta.pt`, or you can directly use the model we trained which are saved at [zenodo](https://zenodo.org/uploads/17728758). The performance of the model is saved at `~/results/`

- Embedding generation and following step: After finishing the step of training or loading the models, run `python embedding_generator.py` to get embeddings of sequences. With the generated embedding, you can move to the step of neighbor search procedure by [neighbor_index_experiments](https://github.com/Shao-Group/MELO-ED/tree/master/neighbor_index_experiments). 


