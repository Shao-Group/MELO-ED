import sys
import pickle
import numpy as np
from scipy.sparse import load_npz

if len(sys.argv) < 3:
    print('Usage: merge_npz2list.py <output-filename> <npz-files> ..\nCombine the input npz files into a list and dump into a pickle file (compatible with evaluate_separate_gapped.py)')
    sys.exit(1)

result = [load_npz(sys.argv[i]) for i in range(2,len(sys.argv))]

with open(sys.argv[1], 'wb') as fout:
    pickle.dump(result, fout)
