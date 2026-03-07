from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies


try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("Missing dependency `numpy`. Install project dependencies.")
    return np


def _unwrap_clip_features(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


class MultiModalSearch:
    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        cache_dir: str | Path = "cache",
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.documents = documents or []
        self.texts: list[str] = []
        self.text_embeddings = None

        if documents:
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]

            np_mod = _require_numpy()
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "clip_text_embeddings.npy"
            meta_path = cache_dir / "clip_text_embeddings.meta.txt"

            meta = f"{model_name}\n{len(self.texts)}\n"
            if cache_path.exists() and meta_path.exists() and meta_path.read_text() == meta:
                self.text_embeddings = np_mod.load(cache_path)
            else:
                all_feats: list[torch.Tensor] = []
                for i in range(0, len(self.texts), batch_size):
                    batch = self.texts[i : i + batch_size]
                    inputs = self.processor(
                        text=batch, padding=True, truncation=True, return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = self.model.get_text_features(**inputs)
                        feats = _unwrap_clip_features(out)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    all_feats.append(feats.detach().cpu())

                feats_tensor = torch.cat(all_feats, dim=0)
                self.text_embeddings = feats_tensor.numpy()
                np_mod.save(cache_path, self.text_embeddings)
                meta_path.write_text(meta)

    def embed_image(self, image_fpath):
        with Image.open(image_fpath) as img:
            img = img.convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            feats = _unwrap_clip_features(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().numpy()
    
    def search_with_image(self, image_fpath, limit = 5):
        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings not initialized; pass documents into MultiModalSearch(...)")
        image_emb = self.embed_image(image_fpath)

        similarities = []
        for idx, text_emb in enumerate(self.text_embeddings):
            similarities.append((idx, cosine_similarity(image_emb, text_emb)))

        sorted_sims = sorted(similarities, key = lambda x:x[1], reverse=True)
        sorted_sims = sorted_sims[:limit]
        results = []

        for (idx, score) in sorted_sims:
            _doc = self.documents[idx]
            results.append({
                'title': _doc['title'],
                'description': _doc['description'],
                'doc_id': idx,
                'score': score,
            })

        return results

def image_search_command(image_fpath, limit = 5):
    movies = load_movies()
    ms = MultiModalSearch(movies)
    res = ms.search_with_image(image_fpath, limit)
    for i, r in enumerate(res, start=1):
        print(f"{i}. {r['title']} (similarity: {r['score']:.3f})")
        print(f"     {r['description'][:100]}")



def verify_image_embedding(image_fpath):
    ms = MultiModalSearch(documents=None)
    embedding = ms.embed_image(image_fpath)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
