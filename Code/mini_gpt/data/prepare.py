import os
import requests
import numpy as np

# download the dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/earthelf/books/master/poems/%E5%94%90%E8%AF%97%E4%B8%89%E7%99%BE%E9%A6%96.TXT'
    print(f"Downloading {data_url}...")
    with open(input_file_path, 'w', encoding='utf-8') as f:
        response = requests.get(data_url)
        #Version1:some files might be gbk, try text first
        #text = response.content.decode('utf-8', errors='ignore')
        #f.write(text)
        # Version2:明确指定使用 gbk 进行解码，若遇到个别杂质字符用 ? 代替（replace）而不是直接无视丢弃
        #text = response.content.decode('gbk', errors='replace')
        #f.write(text)
        #乱码问题是因为原文本的编码格式导致的。普通的 utf-8 和简单的 gbk 在处理某些老的文本资源时会遇到无法正确解析的字符，导致后面所有的文本都被当做乱码。
        #通过 gb18030 编码重新读取并解码了原数据。这是解决中文古籍或旧文本文档乱码的最佳方式。
        text = response.content.decode('gb18030', errors='ignore')
        f.write(text)
with open(input_file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

import re
# 1. 干掉所有的数字编号和诗名/作者（位于数字同一行，比如：062白居易：谷口别）
raw_text = re.sub(r'^\d{3}.*?\n', '', raw_text, flags=re.MULTILINE)
# 2. 干掉目录格式（比如：卷一、五言古诗 . . . 001）
raw_text = re.sub(r'^.*卷[一二三四五六].*\n', '', raw_text, flags=re.MULTILINE)
raw_text = re.sub(r'^.*乐府.*\.\s*\..*\n', '', raw_text, flags=re.MULTILINE)
# 3. 删除连续多余的空行
raw_text = re.sub(r'\n\s*\n', '\n', raw_text)

print(f"Length of raw dataset in characters: {len(raw_text):,}")

# extract unique characters to build vocabulary
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# create the splits
data = np.array(encode(raw_text), dtype=np.uint16)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# save to bin files
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save vocabulary for generation
import pickle
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)
