import os
# used to save pre-computed image vectors from clip stage
import pickle
from typing import Protocol
import torch
import numpy as np
import torchvision.transforms.functional as TF


class Retriever(Protocol):
    name: str
    def build_index(self, images: list[torch.Tensor], poses: list[torch.Tensor], cache_dir: str, dataset_name: str, device: str = "cuda") -> None:
        ...
    def query(self, image: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        ...

class ClipRetriever:
    """
    This is the implementation of CLIP that takes an image as input, and returns the final noramlised vector
    """ 
    def __init__(self, device: str = "cuda"):
        # Initialise the basic variables
        from transformers import CLIPProcessor, CLIPModel
        self.name = "clip"
        self.device = device
        self.model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.index_features = None
        self.index_poses = None

    def _get_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated feature extraction bypassing CPU and PIL conversions.
        """
        # 1. Convert HWC to CHW and drop the alpha channel if present
        img_chw = image.permute(2, 0, 1)[:3]
        
        # 2. Resize directly on the GPU to 224x224
        img_resized = TF.resize(img_chw, [224, 224], antialias=True)
        
        # 3. Apply standard CLIP normalization constants
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        img_norm = TF.normalize(img_resized, mean, std)
        
        # 4. Add batch dimension -> [1, 3, 224, 224]
        pixel_values = img_norm.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model.get_image_features(pixel_values=pixel_values)
            
        return feat / feat.norm(dim=-1, keepdim=True)

    def build_index(self, images: list[torch.Tensor], poses: list[torch.Tensor], cache_dir: str, dataset_name: str) -> None:
        """
        Create a cache of the encodings of the ground truth images. Only create if cache doesn't exist
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{dataset_name}_{self.name}_index.pkl")
        
        if os.path.exists(cache_path):
            print(f"  [{self.name}] Loading cached index from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.index_features = data["features"].to(self.device)
                self.index_poses = data["poses"]
            return

        print(f"  [{self.name}] Building index for {len(images)} images...")
        feats = []
        for img in images:
            feats.append(self._get_features(img))
        
        self.index_features = torch.cat(feats, dim=0)
        self.index_poses = poses
        
        with open(cache_path, "wb") as f:
            pickle.dump({"features": self.index_features.cpu(), "poses": self.index_poses}, f)

    def query(self, image: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Take the query image -> get normalized 1D feature vector -> compute cosine similarities against indexed database features via matrix multiplication -> return top-k matching image indices and similarity scores.
        """
        if self.index_features is None:
            raise RuntimeError("Index not built. Call build_index first.")
        # Get CLIP Embeddings
        query_feat = self._get_features(image)
        # Get photos that are most similar
        sims = (query_feat @ self.index_features.T).squeeze(0)
        # Get the training poses that are most similar
        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(self.index_poses)))
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]

class MegaLocRetriever:
    def __init__(self, device: str = "cuda"):
        import torchvision.transforms as tfm
        self.name = "megaloc"
        self.device = device
        
        print(f"  [{self.name}] Loading MegaLoc model via torch.hub...")
        # MegaLoc is pulled directly from the author's repository
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(self.device)
        self.model.eval()

        # Official MegaLoc preprocessing: ToTensor, ImageNet normalization, 322x322 resize
        self.transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tfm.Resize(size=[322, 322], antialias=True),
        ])
        
        self.index_features = None
        self.index_poses = None

    def _get_features(self, image: torch.Tensor) -> torch.Tensor:
        from PIL import Image
        import numpy as np

        # detach images from pytorch history
        img_np = image.detach().cpu().numpy()
        if img_np.dtype in [np.float32, np.float64]:
            img_np = (img_np[..., :3] * 255).clip(0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_np)
        
        # Transform and add batch dimension
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            feat = self.model(input_tensor)
        
        # MegaLoc outputs L2-normalized descriptors (shape: [1, 8448]), 
        # but we enforce normalization here to guarantee safe cosine similarity matching.
        return feat / feat.norm(dim=-1, keepdim=True)

    def build_index(self, images: list[torch.Tensor], poses: list[torch.Tensor], cache_dir: str, dataset_name: str) -> None:
        import os
        import pickle
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{dataset_name}_{self.name}_index.pkl")
        
        if os.path.exists(cache_path):
            print(f"  [{self.name}] Loading cached index from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.index_features = data["features"].to(self.device)
                self.index_poses = data["poses"]
            return

        print(f"  [{self.name}] Building index for {len(images)} images...")
        feats = []
        for img in images:
            feats.append(self._get_features(img))
        
        self.index_features = torch.cat(feats, dim=0)
        self.index_poses = poses
        
        with open(cache_path, "wb") as f:
            pickle.dump({"features": self.index_features.cpu(), "poses": self.index_poses}, f)

    def query(self, image: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        if self.index_features is None:
            raise RuntimeError("Index not built. Call build_index first.")
        
        query_feat = self._get_features(image)
        
        # Dot product between normalized vectors gives cosine similarity
        sims = (query_feat @ self.index_features.T).squeeze(0)
        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(self.index_poses)))
        
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]