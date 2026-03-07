from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

class MultiModalSearch:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_image(self, image_fpath):
        with Image.open(image_fpath) as img:
            img = img.convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            if isinstance(out, torch.Tensor):
                feats = out
            elif hasattr(out, "image_embeds") and out.image_embeds is not None:
                feats = out.image_embeds
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                feats = out.last_hidden_state.mean(dim=1)
            else:
                raise TypeError(f"Unexpected CLIP output type: {type(out)}")
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().numpy()

def verify_image_embedding(image_fpath):
    ms = MultiModalSearch()
    embedding = ms.embed_image(image_fpath)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
