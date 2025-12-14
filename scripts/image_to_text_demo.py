import os
import pickle
import torch
from PIL import Image

from clip_retrieval.models.clip_wrapper import CLIPWrapper


def main():
    clip = CLIPWrapper()

    # 载入“文本库”（这里用 captions 作为文本库）
    text_features = torch.load("clip_retrieval/outputs/text_features.pt")  # [N, D]
    with open("clip_retrieval/outputs/captions.pkl", "rb") as f:
        captions = pickle.load(f)

    print("✅ Image→Text 检索已加载。输入图片路径（相对或绝对路径），输入 q 退出。")
    print("例如：clip_retrieval/data/images/005.jpg")

    while True:
        img_path = input("ImagePath> ").strip()
        if img_path.lower() in ["q", "quit", "exit"]:
            break
        if not img_path:
            continue

        if not os.path.exists(img_path):
            print(f"❌ 找不到文件：{img_path}\n")
            continue

        # 1) 读图 + 预处理
        img = clip.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)

        # 2) 编码图像向量
        img_feat = clip.encode_images(img)  # [1, D]

        # 3) 与所有文本向量做相似度，取 TopK
        sims = (img_feat @ text_features.T).squeeze(0)  # [N]
        topk = 5
        values, indices = torch.topk(sims, k=min(topk, sims.numel()))

        print(f"\nTop-{topk} captions：")
        for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            print(f"{rank}. score={score:.4f} | {captions[idx]}")
        print()


if __name__ == "__main__":
    main()
