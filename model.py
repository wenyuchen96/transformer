import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    """
    Single head of self-attention

    The formula for scaled dot-product attention is:
    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(self.config.n_embd, self.config.head_size, bias=False)
        self.query = nn.Linear(self.config.n_embd, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.n_embd, self.config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, x):
        """
        input size: (batch_size, block_size, n_embd)
        output size: (batch_size, block_size, head_size)
        """
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)
        "softmax( (Q @ K.T) / sqrt(d_k) )"
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        "softmax() @ V"
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    "Multiple SingleHeadAttention in parallel"
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([SingleHeadAttention(self.config) for _ in range(self.config.n_head)])
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config.n_embd, 4 * self.config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * self.config.n_embd, self.config.n_embd),
            nn.Dropout(self.config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    "Transformer block: MultiHeadAttention + FeedForward + LayerNorm"
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sa = MultiHeadAttention(self.config)
        self.ffwd = FeedForward(self.config)
        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.ln2 = nn.LayerNorm(self.config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks = nn.Sequential(*[Transformer(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size) # language model head; map back to the vocabulary space

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C) get token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device)) # (T,C) get positional embeddings
        x = tok_emb + pos_emb # (B,T,C) add token and positional embeddings
        x = self.blocks(x) # (B,T,C) pass through transformer blocks
        x = self.ln_f(x) # (B,T,C) apply final layer norm
        logits = self.lm_head(x) # (B,T,vocab_size) get logits for next token prediction

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class Util(nn.Module):
    """Helper functions"""
    def __init__(self, text, config):
        super().__init__()
        self.config = config
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.config.vocab_size = self.vocab_size
        self.train_data, self.val_data = self.train_test_split(self.config.train_split_percentage)
    
    def encode(self, string):
        stoi = {ch:i for i,ch in enumerate(self.chars)}
        return [stoi[c] for c in string]
    
    def decode(self, list):
        itos = {i:ch for i,ch in enumerate(self.chars)}
        return ''.join([itos[i] for i in list])
    
    def train_test_split(self, train_percentage):
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(train_percentage*len(data))
        return data[:n], data[n:]
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
    
class EstimateLoss(nn.Module):
    
    def __init__(self, model, config, util):
        super().__init__()
        self.util = util
        self.model = model
        self.config = config
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.util.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

# hyperparameters
class Config:
    batch_size = 64 # amount of indepedent sequences processed in parallel
    block_size = 256 # maximum context length for predictions
    max_iters = 5000 # maximum number of steps to train
    eval_interval = 500 #evaluate performance every 100 steps
    learning_rate = 3e-4 # learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    eval_iters = 200 #average loss over 10 batches when evaluating
    n_embd = 384 # embedding dimension
    n_head = 6 # number of attention heads in multi-head attentions
    head_size = n_embd // n_head # size of each attention head
    n_layer = 6 # number of transformer blocks stacked in the model
    dropout = 0.2 # randomly zero out 20% of neurons during training to prevent overfitting
    train_split_percentage = 0.9 # percentage of data to use for training
    vocab_size = None # set dynamically based on the data

    torch.manual_seed(1337)

if __name__ == '__main__':
    # reading data inputs
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    config = Config()
    util = Util(text, config)

    model = GPT(config)
    m = model.to(config.device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = EstimateLoss(model, config, util).estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = util.get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(util.decode(model.generate(context, max_new_tokens=500)[0].tolist()))