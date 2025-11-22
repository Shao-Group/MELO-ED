import sys
import os

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import faiss

# return a list of k hnsw indices
def build_hnsw(ebd: np.ndarray) -> list:
    n, k, m = ebd.shape
    idx = [faiss.IndexHNSWFlat(m, 32) for _ in range(k)]
    for i in range(k):
        idx[i].add(ebd[:, i, :])
    return idx


# compute top K results from each index
# report pairs with distances <= delta*delta (FAISS index with squared L2 distance)
# store results in the given list of num_indices scipy.sparse.lil_matrix,
# each of shape num_query * num_base and of dtype np.bool_
# if ava (base == query), only report pairs (i, j) if j > i; otherwise report all suitable pairs.
def query_hnsw_delta_separate(nq: np.ndarray, indices: list, K: int, result: list, ava: bool):
    delta_squared = delta * delta
    n, k, m = nq.shape
    assert k == len(indices)

    for i in range(k):
        dist, idx = indices[i].search(nq[:, i, :], K)
        in_range = np.where(dist <= delta_squared)
        for x, y in zip(*in_range):
            z = idx[x, y]
            if not ava or z > x:
                result[i][x, idx[x, y]] = True


import time
import h5py

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: hnsw_knn.py embedding.hdf5 K outputDir\nIf the hdf5 has a single embedding matrix under the key "data", output a list of num_ebd scipy.sparse.csr_matrix of type np.bool_ of shape data x data in a pickle file. Otherwise assume the hdf5 file contains two embedding matrices "embed_a" and "embed_b", output a list of num_ebd scipy.sparse.csr_matrix of type np.bool_ of shape embed_a x embed_b.\n')
        sys.exit(1)

    K = int(sys.argv[2]) # report K nearest neighbors
    delta = 0.0
    output_filename = sys.argv[3] + "/" + os.path.splitext(os.path.basename(sys.argv[1]))[0] + f".{K}NN.separate.pkl"

    import re
    match = re.search('_([.0-9]+)delta_', sys.argv[1])
    if match is None:
        sys.stderr.write(f'Error: cannot parse parameters delta from {sys.argv[1]}\n')
        sys.exit(1)
    else:
        delta = float(match.expand('\g<1>'))
        print(f'delta={delta}', flush=True)
        
    start_time = time.time()
    print('Loading embedding matrix ...', end=' ', flush=True)
    with h5py.File(sys.argv[1], 'r') as fin:
        if len(fin.keys()) == 1:
            ava = True
            base = query = fin['data'][:]
        else:
            ava = False
            base = fin['embed_b'][:]
            query = fin['embed_a'][:]

    print('done\nBuilding HNSW index ...', end=' ', flush=True)

    indices = build_hnsw(base)
    print('done\nQuerying indices ...', end=' ', flush=True)

    num_base, num_ebd, ebd_dim = base.shape
    num_query, _, _ = query.shape
    
    result = [lil_matrix((num_query, num_base), dtype=np.bool_) for _ in range(num_ebd)]
    query_hnsw_delta_separate(query, indices, K, result, ava)
    import pickle
    with open(output_filename, "wb") as fout:
         pickle.dump([m.tocsr() for m in result], fout)

    print(f'done\nResults written to {output_filename}')
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time:.6f} seconds')
