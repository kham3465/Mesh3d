"""Inference utilities: build gallery and infer an image.
"""
import argparse
import os
import sys

# Thêm thư mục gốc vào sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import numpy as np
from PIL import Image
from recognition.models import EmbeddingNet
from recognition.dataset import default_transforms, build_gallery_embeddings
from FaceBoxes.FaceBoxes import FaceBoxes
import cv2


def infer_image(model, gallery, img_path, face_detector=None, device='cpu'):
    """Infer best matching label for image (img_path).
    If image contains full scene, attempt to detect first face with FaceBoxes and crop it.
    gallery is dict label->tensor or ndarray on same device (or will be moved).
    Returns (best_label, best_sim_float).
    """
    transform = default_transforms(112)

    # try to open image and detect face
    pil = Image.open(img_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    try:
        if face_detector is not None:
            dets = face_detector(img_cv)
    except Exception:
        dets = []

    if dets:
        # use first detection
        b = dets[0]
        x1, y1, x2, y2 = map(int, b[:4])
        h, w = img_cv.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 > x1 and y2 > y1:
            crop_cv = img_cv[y1:y2, x1:x2]
            pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))

    x = transform(pil).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        emb = model(x)
    emb = emb.squeeze(0)  # keep on device

    # cosine similarity with gallery (ensure gallery items moved to same device)
    best = None
    best_sim = -1.0
    if not gallery:
        return None, best_sim

    for label, gemb in gallery.items():
        # convert/load gallery item to tensor on same device
        if isinstance(gemb, torch.Tensor):
            g = gemb.to(emb.device)
        else:
            g = torch.as_tensor(np.array(gemb), device=emb.device)
        # normalize
        g = g / (g.norm() + 1e-8)
        sim = float(torch.dot(emb, g).item())
        if sim > best_sim:
            best_sim = sim
            best = label

    return best, best_sim


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gallery_root', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    # load checkpoint to CPU first (robust) then load into model
    ck = torch.load(args.checkpoint, map_location='cpu')
    model = EmbeddingNet()
    ck_model = ck.get('model', ck)
    try:
        model.load_state_dict(ck_model)
    except Exception:
        model.load_state_dict(ck_model, strict=False)
    model.to(device).eval()

    fb = FaceBoxes()
    gallery = build_gallery_embeddings(model, args.gallery_root, device=device)
    label, sim = infer_image(model, gallery, args.image, face_detector=fb, device=device)
    print('Best match:', label, 'score=', sim)