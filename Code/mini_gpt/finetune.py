import os
import re
import torch
import numpy as np
import pickle
from model import GPTConfig, GPT
from utils import log_info

# 1. Hyperparameters for SFT
batch_size = 12
block_size = 128
max_iters = 1000  # 稍微延长SFT的步数，帮助模型更好适应七言等较长的复杂句式
eval_interval = 100
generate_interval = 200
learning_rate = 3e-4  # SFT的初始学习率通常低于预训练
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

data_dir = os.path.join(os.path.dirname(__file__), 'data')
input_file = os.path.join(data_dir, 'input.txt')
ckpt_path = os.path.join(os.path.dirname(__file__), 'mini_gpt.pt')

# 2. 恢复预训练时的分词库
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
    raise FileNotFoundError("找不到预训练阶段生成的 meta.pkl，请先执行预训练流水线。")

# 3. 动态对齐数据工程：从原始文本中动态抽取并在内存中组装
log_info("正在从预训练原始唐诗中动态抽取并构建对仗句微调数据...")
with open(input_file, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# 匹配古诗标准的 “XXX，YYY。” 句式
matches = re.findall(r'([^，。！？\n]+?)，([^，。！？\n]+?)。', raw_text)

sft_lines = []
for m in matches:
    # 稍微做一点数据清洗，保证长度属于正常诗词范畴 (如五言、七言)
    if 4 <= len(m[0]) <= 7 and len(m[0]) == len(m[1]):
        sft_lines.append(f"前句：{m[0]}。续写：{m[1]}。")

log_info(f"成功动态提取了 {len(sft_lines):,} 条形如『前句：...续写：...』的微调指令数据。")

# 顺便导出到本地文件，以便项目审阅
sft_dir = os.path.join(os.path.dirname(__file__), 'sft_data')
os.makedirs(sft_dir, exist_ok=True)
with open(os.path.join(sft_dir, 'train_sft.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(sft_lines[:int(len(sft_lines)*0.9)]))
with open(os.path.join(sft_dir, 'test_sft.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(sft_lines[int(len(sft_lines)*0.9):]))

# 4. 构建监督微调所用的张量序列
sft_text = '\n'.join(sft_lines)
sft_data_encoded = np.array(encode(sft_text), dtype=np.uint16)
n = len(sft_data_encoded)
train_data = sft_data_encoded[:int(n*0.9)]
val_data = sft_data_encoded[int(n*0.9):]

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

# 5. 模型初始化与预训练权重载入
config = GPTConfig(vocab_size=vocab_size, block_size=block_size, dropout=0.2)
model = GPT(config)

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    log_info(f"成功加载预训练权重: {ckpt_path}")
else:
    log_info("警告: 未发现预训练权重，模型将从头开始随机微调（SFT效果会很差）。推荐先运行 train.py 进行预训练。")

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

log_info("=================== 开始监督微调 (SFT) ===================")

# 6. 微调主循环
for iter in range(max_iters + 1):
    
    if iter % eval_interval == 0 or iter == max_iters:
        losses = estimate_loss()
        log_info(f"SFT step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    if iter > 0 and iter % generate_interval == 0:
        model.eval()
        # 同时丢出五言和七言的测试用例进行对比
        test_contexts = [
            "前句：明月出天山。续写：",
            "前句：北风卷地白草折。续写："
        ]
        
        for context in test_contexts:
            idx = torch.tensor([encode(context)], dtype=torch.long, device=device)
            # SFT时的 max_new_tokens 设置得稍大些，并在输出时做阶段截断
            generated_idx = model.generate(idx, max_new_tokens=30)
            generated_text = decode(generated_idx[0].tolist())
            
            # 截断策略：模型会习惯性根据训练数据继续往下吐出“\n前句：XXX”，我们在这里做个截断只看当前这一句
            prompt_len = len(context)
            new_text = generated_text[prompt_len:]
            if '。' in new_text:
                new_text = new_text.split('。')[0] + '。'
            elif '\n' in new_text:
                new_text = new_text.split('\n')[0]
                
            final_output = context + new_text
            log_info(f"--- Step {iter} ---\n{final_output}")
        log_info("----------------------------------\n")
        model.train()

    # training
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

log_info("监督微调完成！你已成功实现了规范化的古诗生成。")
