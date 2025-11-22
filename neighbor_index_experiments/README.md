# Neighbor Index Experiment
This directory contains all scripts needed to reproduce the neighbor search experiments, comparing MELO-ED with the HNSW index against other methods including minimizer, CGK embedding with the bit sampling LSH, and our [learned LSB functions](https://github.com/Shao-Group/lsb-learn). A small sample data with 1,000 pairs of length-20 sequences are also provided, together with their pairwise ground-truth edit distances, LSB vectors, and MELO-ED embeddings.  

## Preparation
```bash
pip install -r requirements.txt
```

## Usage
We use [sample-b.txt](unsafe:data/sequences/sample-b.txt) as the base set $B$ and [sample-a.txt](unsafe:data/sequences/sample-a.txt) as the query set $Q$. Given query thresholds $(d\_1, d\_2)$, the goal is for each sequence $q\\in Q$ to find all sequences $b\\in B$ with edit$(q, b)\\leq d\_1$, while avoiding reporting sequences $x\\in B$ with edit$(q, x)\\geq d\_2$. In our experiments, pairwise edit distances between $Q$ and $B$ have been computed as the evaluation ground-truth, however, this is in general not tractable for large genomic datasets.

### Query with MELO-ED embeddings
Embeddings for both $Q$ and $B$ are generated and stored in the same hdf5 file. For example, to perform a $(1,3)$-sensitive search using [HNSW](unsafe:1,3) 1-near-neighbor:

```bash
python hnsw_knn_heatmap.py 'data/embeddings/embed_sample_20k_40m_(1-3)s_10delta_bd2_10m_bn.hdf5' 1 .
```
generates the resulting file `embed_sample_20k_40m_(1-3)s_10delta_bd2_10m_bn.1NN.separate.pkl'` which can be evaluated by:
```bash
python evaluate_separate_gapped.py data/ground-truth/sample.gt.threshold{1,2}.npz 'embed_sample_20k_40m_(1-3)s_10delta_bd2_10m_bn.1NN.separate.pkl'
```
Note that we used ground-truth files with threshold 1 and 2 to evaluate this $(1,3)$-sensitive search because sample.gt.thresholdX.npz contains all pairs in $Q\\times B$ that has edit distance at most X. In this search, [sample.gt.threshold1.npz](data/ground-truth/sample.gt.threshold1.npz) provides all true positive pairs; whereas any reported pair not in [sample.gt.threshold1.npz](data/ground-truth/sample.gt.threshold2.npz) is a false positive.

### Query with learned LSB functions
Mapping vectors for both $Q$ and $B$ are generated and stored in the same hdf5 file. For example, to perform a $(3,4)$-sensitive search:

```bash
python bucket_by_lsb_heatmap.py 'data/lsb/hash_sample_20k_40m_(3-4)s_10xs.h5' 'lsb_(3-4).result.pkl'
```
Similarly, it can be evaluated by:
```bash
python evaluate_separate_gapped.py data/ground-truth/sample.gt.threshold{3,3}.npz 'lsb_(3-4).result.pkl'
```
Note that since this query is ungapped (i.e., $d_2=d_1+1$), we use [sample.gt.threshold3.npz](data/ground-truth/sample.gt.threshold3.npz) for detecting both true and false positives.

### Query with CGK embeddings
```
Usage: bucket_by_cgk.py <base-file> <query-file> <seq-len> <num-embeddings> <lsh-repeat> <lsh-len> <output-filename>
```
For example, for the length-20 sequences, to generate 20 embeddings with a single LSH per embedding that samples 43 positions (recall that CGK maps a length-20 sequence to a length-60 sequence):
```bash
python bucket_by_cgk.py data/sequences/sample-{b,a}.txt 20 20 1 43 cgk.result.pkl
```
Pay special attention not to misplace the base and query files. Since CGK is not designed for $(d_1, d_2)$-sensitive searches, one has to play with the number of sampled bits of the LSH to find the best performance parameter for a given query setting. To evaluate, for example, for a (1,3)-query:
```bash
python evaluate_separate_gapped.py data/ground-truth/sample.gt.threshold{1,2}.npz cgk.result.pkl 
```

### Query with minimizers
```
Usage: bucket_by_minimizer.py <base-file> <query-file> <k> <output-filename>
```
This implementation relies on the random seeding mechanism of python, so each call to the above function produces results using a different set of minimizers.
To generate 20 minimizers per sequence with $k=9$:
```bash
mkdir mm-k9
for i in {1..20}; do python bucket_by_minimizer.py data/sequences/sample-{b,a}.txt 9 mm-k9/repeat$i.npz; done
```
To evaluate, we first combine the results of these 20 repeats:
```bash
python merge_npz2list.py mm-k9.result.pkl mm-k9/repeat{1..20}.npz
```
Similar to CGK, one needs to play with the parameter $k$ to find the best performance for a given $(d_1, d_2)$ setting. To evaluate, for example, for a (2,3)-query:
```bash
python evaluate_separate_gapped.py data/ground-truth/sample.gt.threshold{2,2}.npz mm-k9.result.pkl
```

