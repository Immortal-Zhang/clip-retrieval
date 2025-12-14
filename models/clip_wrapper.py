import torch
import open_clip

class CLIPWrapper:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = images.to(self.device)
            feats = self.model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def encode_texts(self, texts) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()
