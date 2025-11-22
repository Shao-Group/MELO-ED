import os
import numpy as np
from scipy.sparse import csr_matrix, load_npz
import pickle
import sys

if len(sys.argv) != 4:
    print('Usage: evaluate_separate_gapped.py <ground-truth-le-d1.npz> <ground-truth-le-d2.npz> <predicted-separate.pkl>\nA prediction is considered TP if it is in ground-truth-le-d1; and is considered FP if it is NOT in ground-truth-le-d2. Namely, this is meant to evaluate a (d1, d2+1)-sensitive prediction. Predictions that falls between (d1, d2] are ignored.\nBoth the ground truth and the predictions are scipy.sparse.csr_matrix of dtype np.bool_. The prediction pickle file contains a list of num_embeddings matrices, output TP and FP for gradually combined predictions (1, 2, ..., num_embeddings). ')
    sys.exit(1)

gt_d1 = load_npz(sys.argv[1])
num_gt = gt_d1.count_nonzero()
print(f'ground truth: {num_gt}')

gt_d2 = load_npz(sys.argv[2])
assert(gt_d1.shape == gt_d2.shape)

with open(sys.argv[3], 'rb') as fin:
    prediction = pickle.load(fin)
    
num_embeddings = len(prediction)
cur = csr_matrix(gt_d1.shape, dtype=np.bool_)

for i in range(num_embeddings):
    cur += prediction[i]
    num_cur = cur.count_nonzero()
    tp = cur.multiply(gt_d1) # intersection by elementwise AND
    num_tp = tp.count_nonzero()
    
    non_fp = cur.multiply(gt_d2)
    num_non_fp = non_fp.count_nonzero()
    num_fp = num_cur - num_non_fp

    # recall = num_intersection / num_gt
    # precision = num_intersection / num_cur
    # f1 = 2 * recall * precision / (precision + recall)
    # print(f'{i+1} recall: {recall:.6f} precision: {precision:.6f} f1: {f1:.6f}')
    print(f'{i+1} tp: {num_tp} fp: {num_fp} all: {num_cur}')
