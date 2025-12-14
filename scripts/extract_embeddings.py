import os
import pickle
import torch
from torch.utils.data import DataLoader

from clip_retrieval.data.dataset import ImageTextDataset
from clip_retrieval.models.clip_wrapper import CLIPWrapper


def main():
    clip = CLIPWrapper()

    csv_path = "clip_retrieval/data/captions.csv"
    image_root = "clip_retrieval/data"

    # Dataset 返回：image, caption, img_rel
    dataset = ImageTextDataset(csv_path=csv_path, image_root=image_root, transform=clip.preprocess)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    image_feats_list = []
    text_feats_list = []
    captions_all = []
    image_paths_all = []

    for images, captions, img_paths in loader:
        # 图像向量
        img_feats = clip.encode_images(images)  # [B, D]
        image_feats_list.append(img_feats)

        # 文本向量（用 captions 直接编码）
        txt_feats = clip.encode_texts(list(captions))  # [B, D]
        text_feats_list.append(txt_feats)

        captions_all.extend(list(captions))
        image_paths_all.extend(list(img_paths))

    image_features = torch.cat(image_feats_list, dim=0)  # [N, D]
    text_features = torch.cat(text_feats_list, dim=0)    # [N, D]

    os.makedirs("clip_retrieval/outputs", exist_ok=True)

    torch.save(image_features, "clip_retrieval/outputs/image_features.pt")
    torch.save(text_features, "clip_retrieval/outputs/text_features.pt")

    with open("clip_retrieval/outputs/captions.pkl", "wb") as f:
        pickle.dump(captions_all, f)

    with open("clip_retrieval/outputs/image_paths.pkl", "wb") as f:
        pickle.dump(image_paths_all, f)

    print("✅ saved image_features:", tuple(image_features.shape))
    print("✅ saved text_features :", tuple(text_features.shape))
    print("✅ saved captions      :", len(captions_all))
    print("✅ saved image_paths   :", len(image_paths_all))


if __name__ == "__main__":
    main()
