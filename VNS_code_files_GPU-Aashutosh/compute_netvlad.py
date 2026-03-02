# compute_netvlad.py
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

TILE_DIR = "tiles"
OUT_FILE = "netvlad_tile_descs.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8

print("[INFO] Device:", DEVICE)

# ---------- GeM pooling ----------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=False):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)
        self.eps = eps
    def forward(self, x):
        # x: (B, C, H, W)
        p = self.p if isinstance(self.p, torch.Tensor) else torch.tensor(self.p, device=x.device)
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p)
        x = x.mean(dim=(-1, -2))
        x = x.pow(1.0 / p)
        return x

# ---------- Helper: try to instantiate hloc NetVLAD ----------
def try_hloc_netvlad(device):
    try:
        from hloc.extractors.netvlad import NetVLAD
        import inspect
        sig = inspect.signature(NetVLAD)
        # If NetVLAD accepts no args, instantiate directly; otherwise try passing default conf if available
        if len(sig.parameters) == 0:
            print("[INFO] Found NetVLAD class without args — instantiating.")
            model = NetVLAD().to(device).eval()
            return ("hloc", model, None)
        else:
            # try to find default_conf on module
            import hloc.extractors.netvlad as netconf
            if hasattr(netconf, "default_conf"):
                conf = netconf.default_conf.copy()
                model = NetVLAD(conf).to(device).eval()
                print("[INFO] Found NetVLAD and default_conf — instantiated with conf.")
                return ("hloc", model, None)
    except Exception as e:
        # any failure -> return None
        # print("[DEBUG] hloc NetVLAD import error:", e)
        pass
    return (None, None, None)

# ---------- Build fallback ResNet50+GeM model ----------
def build_resnet_gem(device):
    print("[INFO] Building ResNet50+GeM fallback descriptor model")
    res = models.resnet50(pretrained=True)
    # Remove avgpool and fc: keep layers up to layer4
    layers = [
        res.conv1, res.bn1, res.relu, res.maxpool,
        res.layer1, res.layer2, res.layer3, res.layer4
    ]
    feat_extractor = nn.Sequential(*layers)
    gem = GeM(p=3.0)
    # descriptor model: output L2-normalized vector
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
    return ("resnet_gem", model, 2048)

# ---------- Image transform ----------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ---------- Main ----------
def main():
    # try hloc NetVLAD first
    kind, model, dim = try_hloc_netvlad(DEVICE)
    if kind == "hloc":
        print("[INFO] Using hloc.NetVLAD model for descriptors")
    else:
        kind, model, dim = build_resnet_gem(DEVICE)
        print("[INFO] Using fallback ResNet50+GeM descriptors")

    # prepare tiles list
    fnames = sorted([f for f in os.listdir(TILE_DIR) if f.lower().endswith(".png")])
    if len(fnames) == 0:
        raise RuntimeError(f"No tiles found in {TILE_DIR}")

    tile_keys = []
    vectors = []

    # process in batches
    batch_imgs = []
    batch_keys = []
    for fname in fnames:
        full = os.path.join(TILE_DIR, fname)
        parts = fname.replace(".png", "").split("_")
        try:
            xt, yt = int(parts[1]), int(parts[2])
        except Exception:
            # fallback: if naming differs, use index
            idx = len(tile_keys)
            xt, yt = idx, 0
        tile_keys.append((xt, yt))
        img = Image.open(full).convert("RGB")
        img = img.resize((480, 480))
        t = transform(img)  # C,H,W
        batch_imgs.append(t)
        batch_keys.append((xt, yt))

        if len(batch_imgs) >= BATCH:
            X = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                desc = model(X)  # (B, D)
                desc = desc.cpu().numpy()
                vectors.append(desc)
            batch_imgs = []
            batch_keys = []

    # last partial batch
    if len(batch_imgs) > 0:
        X = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            desc = model(X).cpu().numpy()
            vectors.append(desc)

    vectors = np.vstack(vectors).astype("float32")
    tile_keys_arr = np.array(tile_keys, dtype=object)

    np.savez(OUT_FILE, tile_keys=tile_keys_arr, vectors=vectors)
    print(f"[OK] Saved descriptors → {OUT_FILE}")
    print(f"[INFO] tiles: {len(tile_keys)}, descriptor dim: {vectors.shape[1]} (model: {kind})")

if __name__ == "__main__":
    main()
