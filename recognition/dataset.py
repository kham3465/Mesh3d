import pickle
import torch
import numpy as np
import os

def save_gallery(gallery, fp):
    """Save gallery dict (label -> tensor/ndarray) to disk (pickle)."""
    out = {}
    for k, v in gallery.items():
        if isinstance(v, torch.Tensor):
            arr = v.cpu().numpy()
        else:
            arr = np.array(v)
        out[k] = arr
    d = os.path.dirname(fp)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(fp, 'wb') as f:
        pickle.dump(out, f)

def load_gallery(fp, device='cpu'):
    """Load gallery saved by save_gallery. Return dict label->torch.Tensor on device."""
    if not os.path.exists(fp):
        return {}
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    out = {}
    for k, v in data.items():
        t = torch.from_numpy(v).to(device)
        t = t / (t.norm() + 1e-8)
        out[k] = t
    return out


# New helpers for training / inference
def default_transforms(size=112):
    """Return torchvision transforms for recognition (resize + normalize)."""
    try:
        from torchvision import transforms as T
    except Exception:
        raise ImportError('torchvision is required for default_transforms')
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_gallery_embeddings(model, gallery_root, device='cpu', transform=None, exts=('.jpg', '.jpeg', '.png')):
    """Build gallery dict label->mean_embedding (torch.Tensor on device).

    model: embedding model returning L2-normalized vectors or raw embeddings (will be normalized).
    gallery_root: folder with subfolders per identity containing images.
    device: device for model and output tensors.
    transform: torchvision transform to apply; if None uses default_transforms(112).
    """
    device = torch.device(device)
    model.to(device).eval()
    if transform is None:
        transform = default_transforms(112)

    gallery = {}
    if not os.path.isdir(gallery_root):
        return gallery

    for identity in sorted(os.listdir(gallery_root)):
        id_dir = os.path.join(gallery_root, identity)
        if not os.path.isdir(id_dir):
            continue
        emb_list = []
        for fn in sorted(os.listdir(id_dir)):
            if not fn.lower().endswith(exts):
                continue
            fp = os.path.join(id_dir, fn)
            if not os.path.isfile(fp):
                continue
            try:
                from PIL import Image
                img = Image.open(fp).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model(x)
                e = e.squeeze(0).cpu()
                e = e / (e.norm() + 1e-8)
                emb_list.append(e)
            except Exception:
                continue
        if len(emb_list) == 0:
            continue
        mean = torch.stack(emb_list, dim=0).mean(0)
        mean = mean / (mean.norm() + 1e-8)
        gallery[identity] = mean.to(device)
    return gallery