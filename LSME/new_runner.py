import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import h5py

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from functions import *
from model_loss_train import *
from data_reader_pn_ import data_load_bd


def _expand_slurm_tokens(value):
    """Expand supported Slurm filename tokens in a prefix or suffix."""
    if not value:
        return ''

    token_variables = {
        '%j': ('SLURM_JOB_ID',),
        '%A': ('SLURM_ARRAY_JOB_ID', 'SLURM_JOB_ID'),
        '%a': ('SLURM_ARRAY_TASK_ID',),
    }
    expanded = value
    for token, variable_names in token_variables.items():
        if token not in expanded:
            continue
        replacement = next(
            (os.environ[name] for name in variable_names if os.environ.get(name)),
            None,
        )
        if replacement is None:
            variables = ' or '.join(variable_names)
            raise ValueError(
                f'{token} was requested, but {variables} is not set'
            )
        expanded = expanded.replace(token, replacement)

    if os.sep in expanded or (os.altsep and os.altsep in expanded):
        raise ValueError('output prefixes and suffixes cannot contain path separators')
    if expanded in {'.', '..'}:
        raise ValueError('invalid output prefix or suffix')
    return expanded


def _tagged_filename(stem, extension, prefix='', suffix=''):
    """Build a filename with optional underscore-separated prefix/suffix."""
    components = [component for component in (prefix, stem, suffix) if component]
    return '_'.join(components) + extension


def Training_Evaluation_Parameter_Set(d1, d2, a_, path, df_tr, df_v, df_test,
                                      batch_size, delta, m_dim, num_b,
                                      output_prefix='', output_suffix=''):

    models_path = f'{path}models/'
    os.makedirs(models_path, exist_ok=True)

    results_path = f'{path}results/'
    os.makedirs(results_path, exist_ok=True)

    ID = f'{num_b}k_{m_dim}m_({d1}-{d2})s_{delta}delta'
    encoder_file = os.path.join(
        models_path,
        _tagged_filename(
            f'{Inp_Model_2.__name__}_{ID}', '.pt',
            output_prefix, output_suffix,
        ),
    )
    siamese_file = os.path.join(
        models_path,
        _tagged_filename(
            f'{SiamNNL1.__name__}_{Inp_Model_2.__name__}_{ID}', '.pt',
            output_prefix, output_suffix,
        ),
    )
    results_file = os.path.join(
        results_path,
        _tagged_filename(
            f'loss_acc_{Inp_Model_2.__name__}_{ID}', '.hdf5',
            output_prefix, output_suffix,
        ),
    )

    existing_files = [
        file_path for file_path in (encoder_file, siamese_file, results_file)
        if os.path.exists(file_path)
    ]
    if existing_files:
        formatted_paths = '\n  '.join(existing_files)
        raise FileExistsError(
            'Refusing to overwrite existing output files:\n  '
            f'{formatted_paths}\nChoose a different --output_prefix or --output_suffix.'
        )

    print('Output files:')
    print(f'  encoder: {encoder_file}')
    print(f'  model:   {siamese_file}')
    print(f'  results: {results_file}')

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
    siacnn2 = SiamNNL1(cnnk, flat_dim, out_dim).to(device)
    print(f'{ID} model construct')
    trainer1 = Trainer1(train_a, train_b, train_t, siacnn2, loss0, delta, batch_size)
    print('##########train start###########')
    lr = 0.002 #learning rate, initial = 0.001 
    num_epo = 40 #numbers of epoch
    loss_t = []       
    loss_v = [] 
    for i in range(4):
        lr *= 0.5
        loss1_, loss11_ = trainer1.run(num_epo, lr, valid_a, valid_b, valid_t, m_dim, num_b, device)
        loss_t += loss1_
        loss_v += loss11_ 

        torch.save(cnnk, encoder_file)
        torch.save(siacnn2, siamese_file)
        
    print('##########train end###########')

    print('###########TESTING###############')

    print('breakdown acc')
    print('train: ')
    res, acc_tr = breakdown_acc(eds, d1, d2, siacnn2, acc_count_0, batch_size, delta, m_dim, num_b, device)
    print('acc_train: '+str(acc_tr))
    print('test')
    res_t, acc_t = breakdown_acc(eds_t, d1, d2, siacnn2, acc_count_0, batch_size, delta, m_dim, num_b, device)
    print('acc_test: '+str(acc_t))

    bd_res = []
    for i in sorted(res.keys()):
        bd_res.append([i, res[i]])

    bd_res_t = []
    for i in sorted(res_t.keys()):
        bd_res_t.append([i, res_t[i]])

    with h5py.File(results_file, 'x') as results_handle:
        results_handle.create_dataset('loss_t', data=np.array(loss_t))
        results_handle.create_dataset('loss_v', data=np.array(loss_v))
        results_handle.create_dataset('accs', data=np.array([acc_tr, acc_t]))
        results_handle.create_dataset('bd_train', data=np.array(bd_res))
        results_handle.create_dataset('bd_test', data=np.array(bd_res_t))


#CUDA
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate a Siamese LSH model for fixed-length sequences.',
    )
    parser.add_argument('--n_len', '--N_len', dest='n_len', type=int,
                        required=True, help='Sequence length')
    parser.add_argument('--train_files', type=str, nargs='+', required=True,
                        metavar='FILE',
                        help='One or more training sequence-pair files')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test sequence-pair file')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output directory (models/ and results/ are created inside)')
    parser.add_argument(
        '--output_prefix', type=str, default='',
        help='Prefix added to every output filename. Supports Slurm tokens '
             '%%j (job ID), %%A (array job ID), and %%a (array task ID).',
    )
    parser.add_argument(
        '--output_suffix', type=str, default='',
        help='Suffix added to every output filename. Supports Slurm tokens '
             '%%j (job ID), %%A (array job ID), and %%a (array task ID).',
    )
    parser.add_argument('--d1', type=int, required=True,
                        help='Largest edit distance assigned the positive label')
    parser.add_argument('--d2', type=int, required=True,
                        help='Smallest edit distance assigned the negative label')
    parser.add_argument('--m_dim', type=int, default=40,
                        help='Dimension of each embedding vector')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size')
    parser.add_argument('--num_b', type=int, required=True,
                        help='Number of embedding vectors')
    parser.add_argument('--delta', type=int, default=10)
    parser.add_argument('--rate', type=float, default=0.9,
                        help='Fraction of train/validation data assigned to training')
    parser.add_argument('--num_test', type=int, default=20000,
                        help='Maximum number of test pairs per edit distance')
    parser.add_argument('--num_train_valid', type=int, default=100000,
                        help='Maximum pairs per edit distance and training file')
    parser.add_argument('--max_test_ed', type=int, default=15,
                        help='Largest test edit distance to include '
                             '(default: 15, matching the legacy loaders)')
    args = parser.parse_args()

    if args.n_len <= 0:
        parser.error('--n_len must be positive')
    if args.d1 < 1 or args.d1 >= args.d2:
        parser.error('--d1 must be at least 1 and less than --d2')
    if not 0 < args.rate < 1:
        parser.error('--rate must be between 0 and 1')

    if min(args.m_dim, args.batch_size, args.num_b, args.delta,
           args.num_test, args.num_train_valid) <= 0:
        parser.error('size, count, dimension, and delta arguments must be positive')
    if args.max_test_ed is not None and args.max_test_ed < args.d2:
        parser.error('--max_test_ed must be greater than or equal to --d2')

    out_path = os.path.abspath(os.path.expanduser(args.out_path)) + os.sep
    os.makedirs(out_path, exist_ok=True)
    try:
        output_prefix = _expand_slurm_tokens(args.output_prefix)
        output_suffix = _expand_slurm_tokens(args.output_suffix)
    except ValueError as exc:
        parser.error(str(exc))

    print(f'N={args.n_len}, d1={args.d1}, d2={args.d2}, m_dim={args.m_dim}, '
          f'batch_size={args.batch_size}, num_b={args.num_b}, device={device}')
    print(f'Train files: {args.train_files}')
    print(f'Test file: {args.test_file}')
    print(f'Output path: {out_path}')
    print(f'Output prefix: {output_prefix or "(none)"}')
    print(f'Output suffix: {output_suffix or "(none)"}')

    df_tr, df_v, df_test = data_load_bd(
        args.rate, args.d1, args.d2, args.train_files, args.test_file,
        args.num_test, args.num_train_valid, args.max_test_ed,
    )
    a_ = torch.rand(100, 1, 4, args.n_len).to(device)
    Training_Evaluation_Parameter_Set(
        args.d1, args.d2, a_, out_path, df_tr, df_v, df_test,
        args.batch_size, args.delta, args.m_dim, args.num_b,
        output_prefix, output_suffix,
    )
    
if __name__ == "__main__":
    main()
