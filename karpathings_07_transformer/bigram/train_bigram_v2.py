import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 #  Maximum context length for predictions (how many previous tokens the model sees)
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3 # self-attention models are more sensitive to learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'# Overrides the previous line to force CPU usage
eval_iters = 200
# ------------
n_embd = 32


torch.manual_seed(1337)

# https://www.kaggle.com/code/valkling/monty-python-scripts-database-to-text
with open('All_MP_Scripts_cleaned.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y

    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))# Generate random starting indices for each sequence in batch

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)    # Move data to the appropriate device (CPU/GPU)
    return x, y

@torch.no_grad()# Decorator that disables gradient calculation for efficiency during evaluation
def estimate_loss():
    """
    Calculate average loss over multiple batches for more stable evaluation.
    Returns dictionary with 'train' and 'val' losses.
    """
    out = {}
    model.eval()# Set model to evaluation mode (affects dropout, batch norm, etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()# Set model back to training mode
    return out



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)# what information do i contain?
        self.query = nn.Linear(n_embd, head_size, bias=False)# what am i looking for in other tokens?
        self.value = nn.Linear(n_embd, head_size, bias=False)# what information do i contribute?
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))# pytorch things. As tril is not a parameter, its stored as a buffer


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)    # (B, T, 16)- keys for all tokens
        q = self.query(x)  # (B, T, 16)- queries for all tokens
        # compute attention scores ("affinities")
        # Each query vectors dot product with all key vectors to calculate attention scores
        # If a query and key are aligned (similar direction), they'll produce a high score
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # This gives us a BxTxT tensor where each value at position (i,j) 
        # represents how much token i should attend to token j
        # For each batch, we get a TxT attention matrix (affinity matrix)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)# mask out future positions
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
   """ multiple heads of self-attention in parallel """
   def __init__(self, num_heads, head_size):
       super().__init__()
       # Create multiple attention heads in a list
       self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
       
   def forward(self, x):
       # Run input through each attention head in parallel
       # Concatenate the outputs from all heads along the feature dimension
       # This gives us num_heads * head_size total features per token
       out = torch.cat([h(x) for h in self.heads], dim=-1)
       return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model.
    This model directly predicts the next token based only on the current token,
    with no additional context or memory of previous tokens.
    """
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # This embedding table effectively stores the probability distribution of what comes after each token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table= nn.Embedding(block_size, n_embd) #each position gets its own embedding vector

        #instead one large head we make a small group of heads
        self.selfattn_heads = MultiHeadAttention(4, n_embd//4)
        self.feedforward = FeedFoward(n_embd)
        self.languagemodel_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # Batch size, Sequence length/block_size


        # Get logits (unnormalized probabilities) for next token predictions
        tok_embd = self.token_embedding_table(idx) # Shape: (batch_size, sequence_length, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # Shape: (sequence_length, n_embd)
        
        x= tok_embd + pos_embd #x is the sum(!!!) of the token and position embeddings
        x= self.selfattn_heads(x) # Shape: (batch_size, sequence_length, n_embd)
        x=self.feedforward(x) # B,T,C

        logits = self.languagemodel_head(x) # Shape: (batch_size, sequence_length, vocab_size)

        if targets is None:
            loss = None# If no targets provided, don't compute loss
        else:
            # Reshape logits and targets for computing cross-entropy loss
            B, T, C = logits.shape  # Batch size, Sequence length, Vocabulary size
            logits = logits.view(B*T, C)  # Reshape to (batch_size*sequence_length, vocab_size)
            targets = targets.view(B*T)  # Reshape to (batch_size*sequence_length)
            
            # Cross entropy loss compares predicted distribution with true (one-hot) distribution
            loss = F.cross_entropy(logits, targets)


        return logits, loss

    #commented here, but really used on generate_bigram.py
    def generate(self, idx, max_new_tokens):
        """
        Generate new text by sampling from the model's predicted distribution.
        idx: starting token indices of shape (batch_size, sequence_length)
        max_new_tokens: number of new tokens to generate
        Returns: extended sequence of tokens of shape (batch_size, sequence_length + max_new_tokens)
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #cropt idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last token in each sequence
            logits = logits[:, -1, :] # becomes (B, C)
            # Convert logits to probabilities with softmax
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the probability distribution to get next tokens
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled tokens to the existing sequences
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Directory to save models
os.makedirs("models", exist_ok=True)

best_val_loss = float('inf')  # Initialize best validation loss

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Check if this is the best model so far
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        torch.save(model.state_dict(), "models/bigram_best.pt")
        print(f"Best model saved!  step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss. Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/bigram_last.pt")
print("Training complete!")