import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class ImageTextDataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row["image_path"].strip()
                cap = row["caption"].strip()
                self.samples.append((img_rel, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, caption = self.samples[idx]
        img_path = os.path.join(self.image_root, img_rel)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 返回 img_rel，后面用于把“路径+描述”一起保存并输出
        return image, caption, img_rel
