import torch.nn as nn
from transformers import CLIPModel, AutoModel
from torchvision import transforms as T

class ClipImageEncoder(nn.Module):

    def __init__(self, version="openai/clip-vit-large-patch14", img_size=224):
        super().__init__()

        encoder = CLIPModel.from_pretrained(version)
        encoder = encoder.eval()
        self.encoder = encoder
        self.transform = T.Compose(
            [
                T.Resize(img_size, antialias=True),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def forward(self, image):
        image = self.transform(image)
        embbed = self.encoder.vision_model(image).last_hidden_state
        return embbed


class DinoEncoder(nn.Module):

    def __init__(self, version="facebook/dinov2-large", img_size=224):
        super().__init__()

        encoder = AutoModel.from_pretrained(version)
        encoder = encoder.eval()
        self.encoder = encoder
        self.transform = T.Compose(
            [
                T.Resize(img_size, antialias=True),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def forward(self, image):
        image = self.transform(image)
        embbed = self.encoder(image).last_hidden_state
        print("embbed", embbed.shape)
        return embbed

