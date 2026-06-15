import os
import torch
import numpy as np
import pickle
from model import GPTConfig, GPT

# hyperparameters
batch_size = 12
block_size = 128
max_iters = 2000
eval_interval = 100
generate_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# data loading
data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# attempt to read metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])
else:
    vocab_size = 1000 # default
    encode = lambda s: [0] * len(s)
    decode = lambda l: ''

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# model init
config = GPTConfig(vocab_size=vocab_size, block_size=block_size, dropout=0.2)
model = GPT(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

from utils import log_info, plot_loss

log_info("=================== 开始新的预训练 ===================")
log_info(f"Training on device: {device}")

# Record losses for plotting
train_losses_history = []
val_losses_history = []
eval_steps = []

for iter in range(max_iters + 1):
    
    if iter % eval_interval == 0 or iter == max_iters:
        losses = estimate_loss()
        log_info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_losses_history.append(losses['train'].item())
        val_losses_history.append(losses['val'].item())
        eval_steps.append(iter)
        
    if iter > 0 and iter % generate_interval == 0:
        model.eval()
        context = "床前明月光"
        idx = torch.tensor([encode(context)], dtype=torch.long, device=device)
        generated_idx = model.generate(idx, max_new_tokens=50)
        generated_text = decode(generated_idx[0].tolist())
        log_info(f"\n--- Generation at step {iter} ---\n{generated_text}\n--------------------------\n")
        model.train()

    # training
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Plot the loss curve
plot_loss(train_losses_history, val_losses_history, eval_steps)

# Save pre-trained model weights for fine-tuning
torch.save(model.state_dict(), "mini_gpt.pt")
log_info("预训练模型权重已保存至 mini_gpt.pt")
