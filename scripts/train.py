import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.bigram_model import BigramLanguageModel
from config.configurator import get_config

# Ensure the current working directory is the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

# Load configuration
config = get_config()

# Load your dataset
with open(os.path.join(script_dir, '../data/your_dataset.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Initialize model
model = BigramLanguageModel(vocab_size, config.n_embd, config.n_head, config.n_layer, config.block_size, config.dropout)
model = model.to(config.device)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
for epoch in range(config.max_iters):
    if epoch % config.eval_interval == 0 or epoch == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
