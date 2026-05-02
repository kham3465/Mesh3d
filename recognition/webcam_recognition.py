"""Webcam demo with enrollment:
- detect faces with FaceBoxes
- run TDDFA to get 3DMM params and projected vertices
- extract embedding via recognition model and match to gallery
- draw bbox, label and 3D points on frame
Press 'n' to enroll new person (collect K samples), 's' save snapshot, 'r' rebuild gallery, 'q' quit.
"""
import time
import os
import cv2
import torch
import numpy as np
from FaceBoxes.FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from recognition.models import EmbeddingNet
from recognition.dataset import default_transforms, build_gallery_embeddings, save_gallery, load_gallery

try:
    from utils.functions import crop_img
except Exception:
    crop_img = None

def run(checkpoint, gallery_root, device='cpu', threshold=0.4, cam_id=0, input_size=112, gallery_fp='gallery.pkl'):
    device = torch.device(device)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    ck = torch.load(checkpoint, map_location=device)

    model = EmbeddingNet().to(device)
    model.eval()
    ck_model = ck.get('model', ck)
    try:
        model.load_state_dict(ck_model)
    except Exception:
        model.load_state_dict(ck_model, strict=False)

    face_detector = FaceBoxes()
    tddfa = TDDFA(gpu_mode=(device.type == 'cuda'))
    transform = default_transforms(input_size)

    # Load persisted gallery if exists, otherwise build from folder
    gallery = load_gallery(gallery_fp, device=device)
    if len(gallery) == 0 and os.path.isdir(gallery_root):
        print('Building gallery from folder...')
        gallery = build_gallery_embeddings(model, gallery_root, device=device)
        save_gallery(gallery, gallery_fp)
    print(f'Loaded gallery with {len(gallery)} identities (from {gallery_fp} / {gallery_root})')

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    fps_t0 = time.time()
    fps_count = 0

    print('Press q to quit, s to save a snapshot, r to reload gallery, n to enroll new person')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()

        dets = face_detector(img)
        if len(dets) > 0:
            boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]] for b in dets]
            try:
                params, rois = tddfa(img, boxes)
                verts_lst = tddfa.recon_vers(params, rois, dense_flag=False)
            except Exception:
                params, rois, verts_lst = [], [], []

            for i, (bbox, verts) in enumerate(zip(boxes, verts_lst)):
                x1, y1, x2, y2, score = bbox
                h, w = frame.shape[:2]
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

                # extract crop for recognition (prefer roi crop if available)
                crop = None
                try:
                    if crop_img is not None and rois:
                        # Use the correct ROI for the current face index
                        crop = crop_img(frame, rois[i])
                    else:
                        crop = frame[y1:y2, x1:x2]
                except Exception:
                    crop = frame[y1:y2, x1:x2]

                emb = None
                if crop is not None and crop.size != 0:
                    try:
                        from PIL import Image
                        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        x = transform(pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = model(x).cpu().squeeze(0)
                    except Exception:
                        emb = None

                # match gallery
                best_label = 'Unknown'
                best_sim = -1.0
                if emb is not None and len(gallery) > 0:
                    for label, gemb in gallery.items():
                        sim = float((emb * gemb).sum().item())
                        if sim > best_sim:
                            best_sim = sim
                            best_label = label
                    if best_sim < threshold:
                        best_label = 'Unknown'

                txt = f'{best_label} {best_sim:.3f}' if best_sim >= 0 else best_label
                cv2.putText(img, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # draw projected 3D points (first two coords)
                try:
                    pts2 = verts[:2].T.astype(int)
                    for (px, py) in pts2:
                        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                            cv2.circle(img, (px, py), 1, (0, 0, 255), -1)
                except Exception:
                    pass

        # FPS
        fps_count += 1
        if fps_count >= 5:
            fps = fps_count / (time.time() - fps_t0)
            fps_t0 = time.time()
            fps_count = 0
        else:
            fps = None
        if fps is not None:
            cv2.putText(img, f'FPS: {fps:.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('3D Face Recognition (q to quit)', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            fn = f'webcam_snapshot_{ts}.jpg'
            cv2.imwrite(fn, img)
            print('Saved', fn)
        elif key == ord('r'):
            print('Reloading gallery from folder and file...')
            gallery = load_gallery(gallery_fp, device=device)
            if len(gallery) == 0 and os.path.isdir(gallery_root):
                gallery = build_gallery_embeddings(model, gallery_root, device=device)
                save_gallery(gallery, gallery_fp)
            print(f'Loaded gallery with {len(gallery)} identities')
        elif key == ord('n'):
            # enroll new person
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw() # Ẩn cửa sổ chính của tkinter
            name = simpledialog.askstring("Enrollment", "Enter name/id to enroll:")
            root.destroy()
            
            if name:
                name = name.strip()
            if not name:
                print('Empty name, abort')
            else:
                K = 10
                timeout = 12.0
                collected = []
                print(f'Collecting up to {K} face samples for \"{name}\". Please look at camera.')
                start_t = time.time()
                attempts = 0
                while len(collected) < K and time.time() - start_t < timeout:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        continue
                    dets2 = face_detector(frame2)
                    if not dets2:
                        cv2.imshow('3D Face Recognition (q to quit)', frame2)
                        cv2.waitKey(1)
                        attempts += 1
                        continue
                    bbox2 = dets2[0]
                    x1b, y1b, x2b, y2b = map(int, bbox2[:4])
                    try:
                        params2, rois2 = tddfa(frame2, [[bbox2[0], bbox2[1], bbox2[2], bbox2[3], bbox2[4]]])
                        if crop_img is not None and rois2:
                            crop2 = crop_img(frame2, rois2[0])
                        else:
                            crop2 = frame2[y1b:y2b, x1b:x2b]
                        from PIL import Image
                        pil2 = Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))
                        x_in = transform(pil2).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb2 = model(x_in).cpu().squeeze(0)
                        collected.append(emb2)
                        print(f'Collected {len(collected)}/{K}')
                    except Exception:
                        attempts += 1
                        continue
                    cv2.imshow('3D Face Recognition (q to quit)', frame2)
                    cv2.waitKey(1)
                if len(collected) == 0:
                    print('No valid faces collected, abort enrollment')
                else:
                    mean_emb = torch.stack(collected, dim=0).mean(0)
                    mean_emb = mean_emb / (mean_emb.norm() + 1e-8)
                    gallery[name] = mean_emb.to(device)
                    save_gallery(gallery, gallery_fp)
                    print(f'Enrolled \"{name}\" with {len(collected)} samples. Gallery size={len(gallery)}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gallery', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--cam', type=int, default=0)
    p.add_argument('--gallery_fp', default='gallery.pkl', help='path to persist gallery')
    args = p.parse_args()
    run(args.checkpoint, args.gallery, device=args.device, threshold=args.threshold, 
        cam_id=args.cam, gallery_fp=args.gallery_fp)