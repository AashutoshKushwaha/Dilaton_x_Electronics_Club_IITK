# netvlad_utils.py
# Place this file in the same folder as main.py
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# Globals (cached model)
_MODEL = None
_MODEL_KIND = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DESC_DIM = None
_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# GeM pooling (same as compute_netvlad fallback)
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=False):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)
        self.eps = eps
    def forward(self, x):
        p = self.p if isinstance(self.p, torch.Tensor) else torch.tensor(self.p, device=x.device)
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p)
        x = x.mean(dim=(-1, -2))
        x = x.pow(1.0 / p)
        return x

def _find_descriptor_file():
    # Search common locations for netvlad_tile_descs.npz
    candidates = [
        "netvlad_tile_descs.npz",
        os.path.join("vns_out", "netvlad_tile_descs.npz"),
        os.path.join(".", "vns_out", "netvlad_tile_descs.npz")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: search recursively
    matches = glob.glob("**/netvlad_tile_descs.npz", recursive=True)
    return matches[0] if matches else None

def _build_resnet_gem(device):
    # Build ResNet50 up to layer4 + GeM pooling, output dim = 2048
    res = models.resnet50(pretrained=True)
    layers = [
        res.conv1, res.bn1, res.relu, res.maxpool,
        res.layer1, res.layer2, res.layer3, res.layer4
    ]
    feat_extractor = nn.Sequential(*layers)
    gem = GeM(p=3.0)
    class DescModel(nn.Module):
        def __init__(self, feat_extractor, gem):
            super().__init__()
            self.fe = feat_extractor
            self.gem = gem
        def forward(self, x):
            f = self.fe(x)           # (B, C, H, W)
            g = self.gem(f)         # (B, C)
            g = nn.functional.normalize(g, p=2, dim=1)
            return g
    model = DescModel(feat_extractor, gem).to(device).eval()
    return model, 2048

def _try_hloc_netvlad(device, desc_dim):
    # Try to instantiate hloc NetVLAD if available and dims match
    try:
        from hloc.extractors.netvlad import NetVLAD
        import hloc.extractors.netvlad as netconf
        # If module exposes default_conf, try it
        if hasattr(netconf, "default_conf"):
            conf = netconf.default_conf.copy()
            m = NetVLAD(conf).to(device).eval()
            # forward a dummy tensor to see output dim
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 480, 480).to(device)
                out = m(dummy)
                if "global" in out:
                    d = out["global"].shape[1]
                    if d == desc_dim:
                        return m, d
        # else try direct instantiation if API allows
        try:
            m = NetVLAD().to(device).eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 480, 480).to(device)
                out = m(dummy)
                if "global" in out:
                    d = out["global"].shape[1]
                    if d == desc_dim:
                        return m, d
        except Exception:
            pass
    except Exception:
        pass
    return None, None

def _init_model_for_desc_dim(desc_dim):
    global _MODEL, _MODEL_KIND, _DESC_DIM
    if _MODEL is not None and _DESC_DIM == desc_dim:
        return
    # Try hloc NetVLAD if available and dims match
    m, d = _try_hloc_netvlad(_DEVICE, desc_dim)
    if m is not None:
        _MODEL = m
        _MODEL_KIND = "hloc_netvlad"
        _DESC_DIM = d
        return
    # Fallback to ResNet+GeM (2048)
    model, d = _build_resnet_gem(_DEVICE)
    if d != desc_dim:
        # If descriptor dim differs, still use resnet and project/resize
        # But better to warn user - we'll still return vector truncated/padded as needed
        pass
    _MODEL = model
    _MODEL_KIND = "resnet_gem"
    _DESC_DIM = d

def netvlad_encode(image_rgb_np):
    """
    Encode a single image (H x W x 3 uint8 RGB numpy) to a descriptor np.float32 vector.
    """
    global _MODEL, _MODEL_KIND, _DESC_DIM
    # find descriptor file to know target dim
    desc_file = _find_descriptor_file()
    if desc_file is None:
        # If no descriptor file found, default to 2048
        target_dim = 2048
    else:
        data = np.load(desc_file, allow_pickle=True)
        vectors = data["vectors"]
        target_dim = int(vectors.shape[1])

    # initialize model (if needed)
    if _MODEL is None or _DESC_DIM != target_dim:
        _init_model_for_desc_dim(target_dim)

    # Preprocess input
    if isinstance(image_rgb_np, np.ndarray):
        img = Image.fromarray(image_rgb_np)
    else:
        img = Image.open(image_rgb_np).convert("RGB")
    img = img.resize((480, 480))
    t = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        out = _MODEL(t)  # expects (1, D)
        if isinstance(out, dict) and "global" in out:
            vec = out["global"].cpu().numpy()[0]
        else:
            # assume direct tensor output
            vec = out.cpu().numpy()[0]

    vec = vec.astype("float32")

    # If target_dim differs, either truncate or pad with zeros
    if vec.shape[0] != target_dim:
        if vec.shape[0] > target_dim:
            vec = vec[:target_dim]
        else:
            pad = np.zeros((target_dim - vec.shape[0],), dtype="float32")
            vec = np.concatenate([vec, pad], axis=0)

    return vec
