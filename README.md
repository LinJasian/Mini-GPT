# 📝 从零构建大模型：Mini GPT (基于《唐诗三百首》)

本项目是一个极简主义（纯 PyTorch、无冗余框架）的大语言模型（LLM）教学项目，旨在帮助学习者从零开始理解和构建类似 GPT 的底层架构。项目确保在单机/低算力平台（如笔记本电脑 CPU 或单台消费级显卡）上能够 100% 跑通，并提供完整的可视化验证。

## 📂 项目目录结构

```plaintext
mini_gpt/
├── data/
│   ├── input.txt           # 原始清洗后的训练集《唐诗三百首》文本
│   ├── prepare.py          # 数据下载、正则清洗与分词器 (Tokenizer) 脚本
│   ├── train.bin           # 预训练训练集（ uint16 二进制映射文件）
│   ├── val.bin             # 预训练验证集
│   └── meta.pkl            # 词表和元数据
├── sft_data/
│   ├── train_sft.txt       # 微调训练集（动态对齐组装生成）
│   └── test_sft.txt        # 微调测试集
├── results/
│   ├── training_log.txt    # 打印和记录详细的 Loss 以及文本推演日志
│   └── loss_curve.png      # 由 utils.py 自动生成的损失收敛图表
├── model.py                # 任务一：核心 GPT 模型架构搭建
├── train.py                # 任务二：自监督预训练 (Pre-training) 程序
├── finetune.py             # 任务三：监督微调 (Supervised Fine-Tuning) 程序
├── utils.py                # 辅助工具函数（测绘、日志系统）
└── requirements.txt        # 基础依赖清单
```

## 🧠 核心设计思路

### 1. 架构级极简主义
本项目不依赖于 HuggingFace 的 transformers 等高级封装库。在 `model.py` 中，采用自底向上的方式纯手工构建了 4 个主要核心组件：
*   **`GPTConfig`**：控制参数规模，默认超低开销配置 (`block_size=128`, `vocab_size=自动抽取`, `n_layer=4`, `n_head=4`, `n_embd=128`)。
*   **`CausalSelfAttention`**：使用底层的张量拆分计算 $Q, K, V$，通过下三角矩阵作 Mask 掩码。
*   **`Block`**：标准 Transformer 解码器块，基于 Pre-LayerNorm 架构增强训练稳定性。
*   **`GPT`**：模型主类，包含词嵌入(`wte`)、位置嵌入(`wpe`)、迭代网络与用于文本续写的自回归 `generate` 函数。

### 2. 数据工程与正则净化
我们从无版权的《唐诗三百首》文本入手，并手动实现了一个**字符级 Tokenizer**。对于自然语言处理，最忌讳的是“格式噪声”。
*   在 `prepare.py` 中，引入正则流水线剔除诗歌编号、诗人信息以及目录导航，消除数据“杂质污染”，避免模型过度拟合排版格式。
*   使用 `np.memmap` 机制在 `train.py` 和 `finetune.py` 中实现 O(1) 级别的 Batch 高速随机加载。

### 3. 强正则化防过拟合
由于数据集小（古诗语料几万字）且模型紧凑，如果直接训练极易陷入过拟合并机械吐词。因此我们在网络与优化器内打入了“防过拟合组合拳”：
*   **Dropout**：配置提升至 0.2，分别安插在自注意力后、MLP映射后以及Embedding之后。
*   **Weight Decay**：在 AdamW 优化器中施加了 $0.1$ 的权重衰减，强制要求模型寻找泛用更广的规则，而不是“死记硬背”。

### 4. 监督微调理念 (SFT): 解耦“格式”与“知识”
有别于常规模型直接背诵句子，在 `finetune.py` 中，我们以动态对齐工程的方式提取对仗句式，即时在内存中重组为：
`前句：XXX。续写：YYY。`
*   **激发与规范 (Trigger & Steer)**：SFT 的本质不是学知识（知识在预训练获得），而是激活模型在预训练时学到的深层意象，规范化它的输出格式以对应人类指令。
*   **推理截断**：由于模型会过度模仿长文本排版习惯往下接话，我们在生成结果时加入了动态标点和换行截断，保证其拥有清晰干净的 QA 效果。

## 🚀 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 步骤一：数据构建
爬取古书并构建 Tokenizer 及映射：
```bash
python data/prepare.py
```

### 步骤二：基座模型预训练 (Pre-Training)
将词表进行不断迭代映射，你可以在控制台直接观察到模型从“乱码”到“标点成型”再到“押韵古诗”的进化过程。
```bash
python train.py
```
预训练完成后，将会在目录下出现收敛曲线图与 `mini_gpt.pt` 权重。

### 步骤三：监督微调 (Fine-Tuning)
加载预训练权重，使用动态指令进行训练，使得大模型学会对对子。
```bash
python finetune.py
```
可以在屏幕中观察对于五言诗和七言诗的定向生成成果。

---
*本项目完全遵循极简与高效原则，旨在剥下现代大模型的神秘感，体会万丈高楼平地起的工程美感。*