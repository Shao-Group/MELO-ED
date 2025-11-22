import sys
import numpy as np
from scipy.sparse import lil_matrix
import pickle
import subprocess
from itertools import combinations
    
alphabet_size = 4 # A C G T
filler = "N"
def char2int(x):
    r = ord(x)
    return 3 & ((r >> 2) ^ (r >> 1))

# produce the cgk embedding of s using the provided random function f
# represented as a binary vector of size 3|s|*alphabet_size
# see the variable cgk_random_functions below
def cgk_embedding(s:str, f:np.ndarray) -> np.ndarray:
    i = 0
    n = len(s)
    r = np.full(3 * n, filler)
    for j in range(len(r)):
        if i < n:
            r[j] = s[i]
            i += f[j * alphabet_size + char2int(s[i])]
        else:
            break
    return r # return array for ease of indexing when lsh
    

if __name__ == '__main__':

    if len(sys.argv) != 8:
        print('Usage: bucket_by_cgk.py <base-file> <query-file> <seq-len> <num-embeddings> <lsh-repeat> <lsh-len> <output-filename>\nAssumes the input files have one sequence per line. Generates a list of <num-embeddings> num_query x num_base scipy.sparse matrices of dtype np.bool_ where an entry (i,j) is true if and only the two sequences have at least one similar embedding. The list is stored in a pickle file with name <output-filename>.\nTo determine if two CGK embeddings are similar, the LSH for hamming distance is used: <lsh-repeat> number of random samplings of length <lsh-len> are generated, the embeddings are similar if they share at least one random sampling (at the same repeat index). If base and query are the same file, only report pairs with i < j.')
        sys.exit(1)


    seq_len = int(sys.argv[3])
    num_embd = int(sys.argv[4])
    embd_len = 3 * seq_len

    lsh_repeat = int(sys.argv[5])
    lsh_len = int(sys.argv[6])
    ava = (sys.argv[1] == sys.argv[2])
    output_filename = sys.argv[7]

    rng = np.random.default_rng()
    
    # get number of sequences
    if ava:
        num_base = num_query = int(subprocess.run(['wc', '-l', sys.argv[1]], capture_output=True, text=True).stdout.split()[0])
    else:
        num_base = int(subprocess.run(['wc', '-l', sys.argv[1]], capture_output=True, text=True).stdout.split()[0])
        num_query = int(subprocess.run(['wc', '-l', sys.argv[2]], capture_output=True, text=True).stdout.split()[0])

    # each as a set of functions f_1, ..., f_{embd_len}
    # f_j(char) takes the value at [(j-1)*alphabet_size + char2int(char)]
    cgk_random_functions = [rng.choice([0,1], size=embd_len*alphabet_size) for _ in range(num_embd)]

    lsh_random_functions = [rng.choice(embd_len, size=lsh_len) for _ in range(lsh_repeat)]
    
    # build buckets {subseq_sample(CGK_embedding):list(indices)} on base,
    # one group for each num_embd, in each group, one for each lsh_repeat
    buckets = [[dict() for _ in range(lsh_repeat)] for _ in range(num_embd)]
    with open(sys.argv[1], 'r') as fin:
        idx = 0
        for l in fin:
            seq = l.strip().split()[0]
            seq_cgk_embds = [cgk_embedding(seq, f) for f in cgk_random_functions]
            for i in range(num_embd):
                for j in range(lsh_repeat):
                    #seq_lsh = ''.join(seq_cgk_embds[i][lsh_random_functions[j]])
                    seq_lsh = seq_cgk_embds[i][lsh_random_functions[j]].view(f'<U{lsh_len}')[0]
                    if seq_lsh in buckets[i][j]:
                        buckets[i][j][seq_lsh].append(idx)
                    else:
                        buckets[i][j][seq_lsh] = [idx]

            idx += 1


    results = [lil_matrix((num_query, num_base), dtype=np.bool_) for _ in range(num_embd)]

    # fill the result matrices
    if ava:
        for i in range(num_embd):
            for j in range(lsh_repeat):
                for b in buckets[i][j].values():
                    for x, y in combinations(b, 2): # works since the indices in buckets are sorted by construction
                        results[i][x, y] = True
    else:
        with open(sys.argv[2], 'r') as fin:
            idx = 0
            for l in fin:
                seq = l.strip().split()[0]
                seq_cgk_embds = [cgk_embedding(seq, f) for f in cgk_random_functions]
                for i in range(num_embd):
                    for j in range(lsh_repeat):
                        #seq_lsh = ''.join(seq_cgk_embds[i][lsh_random_functions[j]])
                        seq_lsh = seq_cgk_embds[i][lsh_random_functions[j]].view(f'<U{lsh_len}')[0]
                        if seq_lsh in buckets[i][j]:
                            for x in buckets[i][j][seq_lsh]:
                                results[i][idx, x] = True

                idx += 1

    with open(output_filename, 'wb') as fout:
        pickle.dump([x.tocsr() for x in results], fout)
