import sys
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import time
import h5py

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: bucket_by_lsb.py <lsb-hashcode.hdf5> <output-filename>\nIf the hdf5 has a single hashcode binary matrix under the key "data", output a list of num_hashcode scipy.sparse.csr_matrix of type np.bool_ of shape data x data in a pickle file. Otherwise assume the hdf5 file contains two hashcode binary matrices "hash_a" and "hash_b", output a list of num_hashcode scipy.sparse.csr_matrix of type np.bool_ of shape hash_a x hash_b.\n')
        sys.exit(1)

    start_time = time.time()
    print('Loading hashcode matrix ...', end=' ', flush=True)
    with h5py.File(sys.argv[1], 'r') as fin:
        if len(fin.keys()) == 1:
            ava = True
            base = query = fin['data'][:]
        else:
            ava = False
            base = fin['hash_b'][:]
            query = fin['hash_a'][:]

    print('done\nBuilding buckets with hashcodes ...', end=' ', flush=True)
    
    # get number of sequences
    num_base, num_hashcode, len_hashcode = base.shape
    num_query, _, _ = query.shape

    bits = np.array([2**i for i in range(len_hashcode)])
    base_code = base.dot(bits)

    result = [lil_matrix((num_query, num_base), dtype=np.bool_) for _ in range(num_hashcode)]

    # build num_hashcode sets of buckets {hashcode:list(indices)} on base
    buckets = [dict() for _ in range(num_hashcode)]

    for i in range(num_base):
        for j in range(num_hashcode):
            code = base_code[i, j]
            if code in buckets[j]:
                buckets[j][code].append(i)
            else:
                buckets[j][code] = [i]

    print('done\nQuerying indices ...', end=' ', flush=True)
    # fill the result matrix
    if ava:
        for i in range(num_hashcode):
            for b in buckets[i].values():
                for x, y in combinations(b, 2): # works since the indices in buckets are sorted by construction
                    result[i][x, y] = True
    else:
        query_code = query.dot(bits)
        for i in range(num_query):
            for j in range(num_hashcode):
                code = query_code[i, j]
                if code in buckets[j]:
                    for x in buckets[j][code]:
                        result[j][i, x] = True

    import pickle
    with open(sys.argv[2], "wb") as fout:
         pickle.dump([m.tocsr() for m in result], fout)

    print(f'done\nResults written to {sys.argv[2]}')
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time:.6f} seconds')
