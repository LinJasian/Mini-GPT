import os
import re
import torch
import numpy as np
import pickle
from torch.nn import functional as F
from model import GPTConfig, GPT
from utils import log_info

# ================= 1. 创新微调参数设置 =================
max_sft_iters = 400
eval_interval = 50
learning_rate = 2e-4
temperature = 0.85  # 温度参数：小于1.0适度集中，大于1.0放飞想象，0.85完美兼顾通顺与创新
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
base_dir = os.path.dirname(__file__)

# ================= 2. 载入预训练元数据与词表 =================
data_dir = os.path.join(base_dir, 'data')
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError("请先运行 data/prepare.py 以确立基础词表字典！")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

# ================= 3. 动态数据工程：从语料中自动抽取诗句对 =================
raw_corpus_path = os.path.join(data_dir, 'input.txt')
if not os.path.exists(raw_corpus_path):
    raise FileNotFoundError("未找到原始语料 input.txt！")

with open(raw_corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

# 使用正则抽取所有形如 "五言/七言，五言/七言。" 的标准诗句对，过滤掉目录和各种数字编号
# 比如抽取: "青山隐隐水迢迢，秋尽江南草未凋。"
poem_pairs = re.findall(r'([^，\n。\d]{5,7})，([^，\n。\d]{5,7})。', corpus)
log_info(f"[📊 数据工程] 从原始唐诗中动态抽取出 {len(poem_pairs)} 组规整的对仗句用于对齐训练。")

# 将抽取的诗句转换为规范的 SFT 指令文本流
sft_text_buffer = ""
for front, back in poem_pairs:
    sft_text_buffer += f"前句：{front}。续写：{back}。\n"

sft_data_encoded = torch.tensor(encode(sft_text_buffer), dtype=torch.long)

def get_dynamic_sft_batch():
    block_size = 128
    ix = torch.randint(len(sft_data_encoded) - block_size, (16,)) # Batch Size = 16
    x = torch.stack([sft_data_encoded[i:i+block_size] for i in ix])
    y = torch.stack([sft_data_encoded[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ================= 4. 初始化模型与参数冻结 =================
ckpt_path = os.path.join(base_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError("未找到 ckpt.pt，请确保阶段二预训练已完成！")

checkpoint = torch.load(ckpt_path, map_location=device)
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.to(device)

# 核心策略：冻结绝大部分底座参数，防止灾难性遗忘，只允许顶层进行行为规范调整
for param in model.parameters():
    param.requires_grad = False
for param in model.blocks[-1].parameters(): # 仅解冻最后一层 Block
    param.requires_grad = True
for param in model.lm_head.parameters():     # 解冻分类输出头
    param.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# ================= 5. 温度可控的生成式创新推理函数 =================
@torch.no_grad()
def creative_generate(model, idx, max_new_tokens, temp=1.0):
    """
    带温度(Temperature)控制的自回归生成函数，专为激发大模型创新设计
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ================= 6. 微调循环 =================
log_info("=================== 开始监督微调(SFT) ===================")
for iter in range(max_sft_iters + 1):
    
    if iter % eval_interval == 0 or iter == max_sft_iters:
        model.eval()
        prompt = "前句：明月出天山。续写："
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated_idx = creative_generate(model, idx, max_new_tokens=15, temp=temperature)
        generated_text = decode(generated_idx[0].tolist())
        log_info(f"\n[Step {iter}] SFT Generation (temp={temperature}):\n{generated_text}\n" + "-"*30)
        model.train()

    # training
    xb, yb = get_dynamic_sft_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

log_info("微调完成！")
