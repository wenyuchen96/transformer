import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64 # amount of indepedent sequences processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000 # maximum number of steps to train
eval_interval = 500 #evaluate performance every 500 steps
learning_rate = 3e-4 # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use gpu if available else cpu
eval_iters = 200 #average loss over 200 batches when evaluating
n_embd = 384 # embedding dimension
n_head = 6 # number of attention heads in multi-head attentions
n_layer = # number of transformer blocks stacked in the model
dropout = 0.2 # randomly zero out 20% of neurons during training to prevent overfitting