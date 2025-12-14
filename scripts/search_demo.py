import os
import pickle
import time
import torch

from clip_retrieval.models.clip_wrapper import CLIPWrapper

# 查询增强：把中文短词扩展成更稳的中英混合查询
EXPAND = {
    "金毛": ["金毛犬 狗", "golden retriever dog", "yellow dog", "golden dog face"],
    "金毛犬": ["golden retriever dog"],
    "哈士奇": ["哈士奇 狗 雪地", "husky dog snow"],
    "小猪": ["小猪 猪", "pig piglet"],
    "猪": ["小猪 猪", "pig piglet"],
    "灰猫": ["灰猫 猫", "gray cat"],
    "猫": ["灰猫 猫", "gray cat"],
}

LOG_PATH = "clip_retrieval/outputs/search_log.txt"


def main():
    # 确保输出目录存在（防止删了 outputs 后写日志失败）
    os.makedirs("clip_retrieval/outputs", exist_ok=True)

    clip = CLIPWrapper()

    # 加载索引（embedding）与元数据
    image_features = torch.load("clip_retrieval/outputs/image_features.pt")  # [N, D]
    with open("clip_retrieval/outputs/captions.pkl", "rb") as f:
        captions = pickle.load(f)
    with open("clip_retrieval/outputs/image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)

    print("✅ 检索系统已加载。输入中文查询（例如：灰猫、小猪、金毛、哈士奇）。输入 q 退出。")

    while True:
        query = input("Query> ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            break
        if not query:
            continue

        # 1) 查询增强
        queries = [query] + EXPAND.get(query, [])

        # 2) 多查询向量融合（平均）
        q_feats = clip.encode_texts(queries)                 # [Q, D]
        q_feat = q_feats.mean(dim=0, keepdim=True)           # [1, D]
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)  # 归一化

        # 3) 相似度检索
        sims = (q_feat @ image_features.T).squeeze(0)        # [N]
        topk = 5
        values, indices = torch.topk(sims, k=min(topk, sims.numel()))

        # 4) 打印 + 记录日志（把 TopK 全部写进去）
        print(f"\nTop-{topk}（使用 queries={queries}）：")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{ts}] query={query} | expanded={queries}\n")

            for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                line = f"{rank}. score={score:.4f} | {image_paths[idx]} | {captions[idx]}"
                print(line)
                lf.write(line + "\n")

        print()  # 空行分隔


if __name__ == "__main__":
    main()
