# uses built-in hash() which randomizes its seed for each run

import sys
import numpy as np
from scipy.sparse import lil_matrix, save_npz
import subprocess
from itertools import combinations
    
def gen_minimizer(seq:str, k:int) -> str:
    hash_vals = np.array([hash(seq[i:i+k]) for i in range(len(seq)-k+1)])
    min_idx = np.argmin(hash_vals)
    return seq[min_idx:min_idx+k]

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print('Usage: bucket_by_minimizer.py <base-file> <query-file> <k> <output-filename>\nAssumes the input files have one sequence per line. Generates a num_query x num_base scipy.sparse matrix of dtype np.bool_ where an entry (i,j) is true if and only the two sequences share a minimizer. If base and query are the same file, only report pairs with i < j.')
        sys.exit(1)

    k = int(sys.argv[3])
    ava = (sys.argv[1] == sys.argv[2])
    
    # get number of sequences
    if ava:
        num_base = num_query = int(subprocess.run(['wc', '-l', sys.argv[1]], capture_output=True, text=True).stdout.split()[0])
    else:
        num_base = int(subprocess.run(['wc', '-l', sys.argv[1]], capture_output=True, text=True).stdout.split()[0])
        num_query = int(subprocess.run(['wc', '-l', sys.argv[2]], capture_output=True, text=True).stdout.split()[0])

    result = lil_matrix((num_query, num_base), dtype=np.bool_)

    # build buckets {minimizer:list(indices)} on base
    buckets = dict()
    with open(sys.argv[1], 'r') as fin:
        idx = 0
        for l in fin:
            seq = l.strip().split()[0]
            minimizer = gen_minimizer(seq, k)
            if minimizer in buckets:
                buckets[minimizer].append(idx)
            else:
                buckets[minimizer] = [idx]
            idx += 1

    # fill the result matrix
    if ava:
        for b in buckets.values():
            for i, j in combinations(b, 2): # works since the indices in buckets are sorted by construction
                result[i, j] = True
    else:
        with open(sys.argv[2], 'r') as fin:
            idx = 0
            for l in fin:
                seq = l.strip().split()[0]
                minimizer = gen_minimizer(seq, k)
                if minimizer in buckets:
                    for x in buckets[minimizer]:
                        result[idx, x] = True
                idx += 1

    save_npz(sys.argv[4], result.tocsr())
