Introduction
==============

This is the code of MELO-ED, including the training of a deep Locality Sensitive Multi-Embedding (LSME) function and downstream experiment:

- A framework that can train an (d1, d2)-LSME function with customized parameters: N, length of sequence; k, number of embedding; m dimensions, and through δ by code: new_loss_runner.py. 
- Applying Neighbor Search by KNN: 

Examples
==============
- Environment: python vision >= 3.11

- Data loading by folder: `data`: 
- Model training: `python new_loss_runner.py`
