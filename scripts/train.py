import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.bigram_model import BigramLanguageModel
from config.configurator import get_config

# Load configuration
config = get_config()

# Load your dataset
with open('../data/your_dataset.txt', 'r', encoding='utf-8') as f:
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
model = BigramLanguageModel()
model = model.to(config.device)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

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
