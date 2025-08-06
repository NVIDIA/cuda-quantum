# Train an ADAPT-GPT model
# Based on https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py

out_dir = '{out_dir}'
eval_interval = 250 # keep frequent because we'll overfit. Determines how often standard loss evaluation occurs
eval_iters = 200  #Determines how many batches are used to calculate validation and training loss during model evaluation
log_interval = 10 # don't print too too often
eval_ar_every = 5000 # how often we do approx ratio evaluation (calling ADAPT-QAOA cudaq). Controls how often the model performs domain-specific evaluation using approximation ratio

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = '{dataset}'
gradient_accumulation_steps = 1
batch_size = 64
block_size = {block_size} # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
n_epochs = 70
max_iters = 30000
lr_decay_iters = 30000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

graph_emb_dim = 500 # default for FEATHER graph
use_graph_emb = {use_graph_emb}
pool_type = '{pool_type}'
n_nodes = {n_nodes}
token_seq_round = '{token_seq_round}'  # rounding digits for token sequence
n_samples = 5

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
