# CLIP 图文检索系统（Text↔Image，Query Expansion，含可复现日志）

本仓库基于开源 **CLIP（ViT-B/32）** 实现一个轻量级多模态检索项目，支持 **Text→Image** 与 **Image→Text** 双向检索，并针对中文短查询引入 **Query Expansion（中英混合扩展）+ 多查询向量融合（mean pooling）** 来提升检索稳定性；同时提供日志落盘，便于复现实验与面试展示。

> 数据说明：`data/images/` 含 **50 张 AI 生成图片（约 23.6MB）**，`data/captions.csv` 为对应标注。  
> 运行产物（`outputs/*.pt/*.pkl/*.txt`）**不入库**（体积大/可本地复现），请按下述步骤在本地生成。

---

## 功能特性

- **Text→Image 检索**：输入中文/英文查询，输出 Top-K（`score + image_path + caption`）
- **Image→Text 检索**：输入图片路径，输出 Top-K captions
- **Query Expansion**：对“金毛/哈士奇/小猪/猫”等中文短词做中英混合扩展
- **向量融合**：多条 query embedding 取均值融合后检索（mean pooling）
- **日志可复现**：自动生成 `outputs/search_log.txt`，记录时间戳、原始 query、扩展 queries、Top-K 结果

---

## 目录结构

```text
clip_retrieval/
  data/
    images/                 # AI 生成图片（已入库）
    captions.csv            # 标注：image_path,caption
    dataset.py              # Dataset（返回 image, caption, image_path）
  models/
    clip_wrapper.py         # CLIP 封装（encode_images / encode_texts）
  scripts/
    extract_embeddings.py   # 抽取 image/text features -> outputs/
    search_demo.py          # Text→Image 检索（含 Query Expansion + 日志）
    image_to_text_demo.py   # Image→Text 检索
  outputs/                  # 运行产物（不入库，本地生成）
    image_features.pt
    text_features.pt
    captions.pkl
    image_paths.pkl
    search_log.txt
```

---

## 环境与依赖

- Python 3.10+（macOS Apple Silicon 支持 MPS 推理加速）
- 依赖：`torch`、`open_clip_torch`、`pillow`、`tqdm`

建议在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install open_clip_torch pillow tqdm
```

> 你也可以复用已有 `.venv` 环境，只要能 `import torch, open_clip` 即可。

---

## 数据格式（captions.csv）

`data/captions.csv` 必须是 **英文逗号分隔**，且表头不要写成 `image_path, caption`（caption 前不能有空格）：

```csv
image_path,caption
images/001.jpg,一只灰猫蹲在树干上
images/002.jpg,一只沙皮狗站在草地
...
```

---

## 快速开始（必须按顺序）

### 1）抽取 embedding（生成索引文件）

> 因为 `outputs/` 不入库，首次运行必须先生成 features。

在仓库根目录执行：

```bash
cd clip_retrieval
python -m scripts.extract_embeddings
```

成功后会在 `clip_retrieval/outputs/` 生成：

- `image_features.pt`：图像 embedding（形状 `[N, 512]`）
- `text_features.pt`：文本 embedding（形状 `[N, 512]`）
- `captions.pkl`、`image_paths.pkl`：元数据
-（如果脚本提示 saved 数量为 50，则与本仓库数据一致）

### 2）Text→Image 检索（含 Query Expansion + 日志）

```bash
python -m scripts.search_demo
```

示例输入：`灰猫` / `金毛` / `哈士奇` / `小猪`

输出示例：

```
Top-5（使用 queries=[...]）:
1. score=0.2992 | images/005.jpg | 金毛犬在黄背景微笑
...
```

日志会写入：

- `outputs/search_log.txt`

查看日志尾部：

```bash
tail -n 30 outputs/search_log.txt
```

### 3）Image→Text 检索

```bash
python -m scripts.image_to_text_demo
```

输入图片路径示例：

```
data/images/005.jpg
```

---

## Query Expansion 说明

`search_demo.py` 内置扩展字典（可自行增补），例如：

- 金毛 → `["金毛犬 狗", "golden retriever dog", "yellow dog", "golden dog face"]`
- 哈士奇 → `["哈士奇 狗 雪地", "husky dog snow"]`
- 小猪 → `["小猪 猪", "pig piglet"]`
- 猫/灰猫 → `["灰猫 猫", "gray cat"]`

策略：
1) 原始 query + 扩展 queries 一起编码；  
2) 多条 query embedding 做 mean pooling；  
3) 再归一化后检索，提升中文短词稳定性。

---

## 常见问题

- **为什么克隆下来没有 outputs？**  
  `outputs/` 是运行产物（特征、日志），为了控制仓库体积默认不入库；请先运行 `extract_embeddings` 本地生成。

- **中文短词为什么会漂移？**  
  与预训练语料覆盖、短词歧义有关。通过中英混合扩展 + 向量融合可以明显改善一致性。

---

## 可扩展方向

- 引入 **Faiss/ANN** 做大规模向量检索加速
- 加入 **Recall@K / mAP** 等离线评估指标
- 将 Query Expansion 规则外置为配置文件，或加入自动同义词/翻译扩展
- 扩大数据规模并支持增量索引更新

---

## 许可与说明

- 图片为 AI 生成数据，用于项目演示与复现；如需替换为自定义数据集，请更新 `data/images/` 与 `data/captions.csv`。
