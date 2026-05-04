"""Webcam demo with enrollment:
- detect faces with FaceBoxes
- run TDDFA to get 3DMM params and projected vertices
- extract embedding via recognition model and match to gallery
- draw bbox, label and 3D points on frame
Press 'n' to enroll new person (collect K samples), 's' save snapshot, 'r' rebuild gallery, 'q' quit.
"""
import time
import os
import sys

# Thêm thư mục gốc vào sys.path để tìm thấy FaceBoxes, TDDFA và recognition
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import cv2
import torch
import numpy as np
import yaml
from FaceBoxes.FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from recognition.models import EmbeddingNet
from recognition.dataset import default_transforms, build_gallery_embeddings, save_gallery, load_gallery

try:
    from utils.functions import crop_img
except Exception:
    crop_img = None

def run(checkpoint, gallery_root, config_path, device='cpu', threshold=0.6, cam_id=0, input_size=112, gallery_fp='gallery.pkl', use_onnx=False, frame_skip=3):
    device = torch.device(device)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    # Sử dụng ONNX Runtime nếu file có đuôi .onnx hoặc được yêu cầu
    onnx_session = None
    model = None
    if checkpoint.endswith('.onnx') or use_onnx:
        import onnxruntime as ort
        print(f"Loading ONNX model from {checkpoint}...")
        # Sử dụng CPU với số luồng tối ưu
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        onnx_session = ort.InferenceSession(checkpoint, sess_options, providers=['CPUExecutionProvider'])
    else:
        print(f"Loading PyTorch model from {checkpoint}...")
        ck = torch.load(checkpoint, map_location=device)
        model = EmbeddingNet().to(device)
        model.eval()
        ck_model = ck.get('model', ck)
        try:
            model.load_state_dict(ck_model)
        except Exception:
            model.load_state_dict(ck_model, strict=False)

    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)

    face_detector = FaceBoxes()
    tddfa = TDDFA(gpu_mode=(device.type == 'cuda'), **cfg)
    transform = default_transforms(input_size)

    # Load persisted gallery if exists, otherwise build from folder
    gallery = load_gallery(gallery_fp, device=device)
    if len(gallery) == 0 and os.path.isdir(gallery_root):
        print('Building gallery from folder...')
        # Sử dụng onnx_session nếu có, nếu không dùng model PyTorch
        gallery = build_gallery_embeddings(onnx_session if onnx_session else model, gallery_root, device=device)
        save_gallery(gallery, gallery_fp)

    # Khởi tạo gallery dạng Tensor để tính toán similarity cực nhanh (vector hóa)
    gallery_names = []
    gallery_tensor = None
    if len(gallery) > 0:
        gallery_names = list(gallery.keys())
        gallery_tensor = torch.stack([v.to(device) for v in gallery.values()])
    
    print(f'Loaded gallery with {len(gallery_names)} identities (from {gallery_fp} / {gallery_root})')

    # Trên Windows, sử dụng cv2.CAP_DSHOW giúp khởi động camera ổn định hơn
    if os.name == 'nt':
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print('Cannot open webcam')
        return

    # Thiết lập camera để tránh bị tối trên Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Xóa các dòng cap.set gây lỗi grab frame trên Windows
    # Nếu cam vẫn tối, hãy kiểm tra ánh sáng phòng hoặc gạt che cam vật lý
    print('Warming up camera...')

    for _ in range(30):
        cap.read()

    fps_t0 = time.time()
    fps_count = 0
    frame_idx = 0
    
    # Biến lưu trữ kết quả của frame trước để hiển thị khi skip
    last_results = [] # List of {bbox, label, sim, verts}

    print('Press q to quit, s to save a snapshot, r to reload gallery, n to enroll new person')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame
        frame_idx += 1
        
        # Tăng tốc: Giảm kích thước ảnh khi đưa vào FaceBoxes
        h_orig, w_orig = frame.shape[:2]
        img_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        img = img_small.copy()

        # Chỉ chạy Recognition và 3D Mesh mỗi N frames để tăng FPS
        if frame_idx % frame_skip == 0 or frame_idx < 5:
            dets = face_detector(img)
            current_results = []
            
            if len(dets) > 0:
                # Chỉ lấy top khuôn mặt lớn nhất để tránh nhiễu và tăng FPS
                dets = sorted(dets, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)[:2]
                
                boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]] for b in dets]
                try:
                    params, rois = tddfa(img, boxes)
                    verts_lst = tddfa.recon_vers(params, rois, dense_flag=True)
                except Exception:
                    params, rois, verts_lst = [], [], []

                for i, (bbox, verts) in enumerate(zip(boxes, verts_lst)):
                    x1, y1, x2, y2 = [int(v * 2) for v in bbox[:4]]
                    
                    # Recognition logic
                    crop = None
                    try:
                        if crop_img is not None and rois:
                            scaled_roi = [v * 2 for v in rois[i]]
                            crop = crop_img(frame, scaled_roi)
                        else:
                            crop = frame[max(0,y1):y2, max(0,x1):x2]
                    except Exception:
                        crop = frame[max(0,y1):y2, max(0,x1):x2]

                    emb = None
                    if crop is not None and crop.size != 0:
                        try:
                            from PIL import Image
                            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            input_tensor = transform(pil).unsqueeze(0).numpy()
                            if onnx_session is not None:
                                inputs = {onnx_session.get_inputs()[0].name: input_tensor}
                                emb = torch.from_numpy(onnx_session.run(None, inputs)[0]).squeeze(0)
                            else:
                                with torch.no_grad():
                                    emb = model(torch.from_numpy(input_tensor).to(device)).squeeze(0)
                        except Exception: pass

                    best_label, best_sim = 'Unknown', -1.0
                    if emb is not None and gallery_tensor is not None:
                        # Tính toán similarity vector hóa (cực nhanh)
                        emb_norm = emb / (emb.norm() + 1e-8)
                        sims = torch.matmul(gallery_tensor, emb_norm.to(device))
                        max_sim, max_idx = torch.max(sims, dim=0)
                        best_sim = float(max_sim.item())
                        best_label = gallery_names[max_idx.item()]
                        
                        if best_sim < threshold: best_label = 'Unknown'
                    
                    current_results.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': best_label,
                        'sim': best_sim,
                        'verts': verts
                    })
            last_results = current_results

        # Vẽ kết quả (từ frame hiện tại hoặc frame trước đó)
        for res in last_results:
            x1, y1, x2, y2 = res['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            
            txt = f"{res['label']} {res['sim']:.2f}" if res['sim'] >= 0 else res['label']
            cv2.putText(display_frame, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw 3D points (Hiển thị toàn bộ các điểm mesh)
            try:
                pts2 = (res['verts'][:2].T * 2).astype(int)
                for i_pt in range(len(pts2)): # Vẽ toàn bộ điểm (stride = 1)
                    px, py = pts2[i_pt]
                    if 0 <= px < w_orig and 0 <= py < h_orig:
                        cv2.circle(display_frame, (px, py), 1, (0, 0, 255), -1)
            except Exception: pass

        # FPS
        fps_count += 1
        if fps_count >= 5:
            fps = fps_count / (time.time() - fps_t0)
            fps_t0 = time.time()
            fps_count = 0
        else:
            fps = None
        if fps is not None:
            cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Hiển thị số lượng người hiện có trong Gallery
        gallery_count = len(gallery_names) if gallery_names else 0
        cv2.putText(display_frame, f'Gallery: {gallery_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('3D Face Recognition (q to quit, c to reset)', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            fn = f'webcam_snapshot_{ts}.jpg'
            cv2.imwrite(fn, display_frame)
            print('Saved', fn)
        elif key == ord('r'):
            print('Reloading gallery from folder and file...')
            gallery = load_gallery(gallery_fp, device=device)
            if len(gallery) == 0 and os.path.isdir(gallery_root):
                gallery = build_gallery_embeddings(onnx_session if onnx_session else model, gallery_root, device=device)
                save_gallery(gallery, gallery_fp)
            
            # Cập nhật tensor để nhận diện được ngay
            if len(gallery) > 0:
                gallery_names = list(gallery.keys())
                gallery_tensor = torch.stack([v.to(device) for v in gallery.values()])
            else:
                gallery_names, gallery_tensor = [], None
            print(f'Loaded gallery with {len(gallery)} identities')
        elif key == ord('c'):
            # Tính năng Reset Gallery
            print('Resetting gallery...')
            gallery = {}
            gallery_names = []
            gallery_tensor = None
            if os.path.exists(gallery_fp):
                os.remove(gallery_fp)
            print('Gallery cleared and gallery.pkl removed.')
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
                timeout = 60.0
                interval = 4.0
                last_sample_t = 0
                collected = []
                print(f'Collecting {K} samples for "{name}" over ~40s. Please rotate your head slowly.')
                start_t = time.time()
                attempts = 0
                while len(collected) < K and time.time() - start_t < timeout:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        continue
                    
                    # Luôn hiển thị camera trong lúc đợi để người dùng không thấy bị đứng hình
                    cv2.imshow('3D Face Recognition (q to quit, c to reset)', frame2)
                    cv2.waitKey(1)
                    
                    # Kiểm tra khoảng cách thời gian giữa các lần lấy mẫu
                    if time.time() - last_sample_t < interval:
                        continue

                    dets2 = face_detector(frame2)
                    if not dets2:
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
                        x_in_numpy = transform(pil2).unsqueeze(0).numpy()
                        
                        if onnx_session is not None:
                            inputs2 = {onnx_session.get_inputs()[0].name: x_in_numpy}
                            emb2 = torch.from_numpy(onnx_session.run(None, inputs2)[0]).squeeze(0)
                        else:
                            with torch.no_grad():
                                emb2 = model(torch.from_numpy(x_in_numpy).to(device)).cpu().squeeze(0)
                        collected.append(emb2)
                        last_sample_t = time.time() # Cập nhật mốc thời gian chụp ảnh thành công
                        print(f'Collected {len(collected)}/{K} (Next sample in {interval}s...)')
                    except Exception:
                        attempts += 1
                        continue
                if len(collected) == 0:
                    print('No valid faces collected, abort enrollment')
                else:
                    mean_emb = torch.stack(collected, dim=0).mean(0)
                    mean_emb = mean_emb / (mean_emb.norm() + 1e-8)
                    gallery[name] = mean_emb.to(device)
                    save_gallery(gallery, gallery_fp)
                    
                    # QUAN TRỌNG: Cập nhật lại tensor để máy nhận ra bạn ngay lập tức
                    gallery_names = list(gallery.keys())
                    gallery_tensor = torch.stack([v.to(device) for v in gallery.values()])
                    
                    print(f'Enrolled \"{name}\" with {len(collected)} samples. Gallery size={len(gallery)}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default='configs/mb1_120x120.yml', help='path to tddfa config')
    p.add_argument('--gallery', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=0.6) # Tăng ngưỡng mặc định để tránh nhận diện nhầm
    p.add_argument('--cam', type=int, default=0)
    p.add_argument('--onnx', action='store_true', help='use onnxruntime for inference')
    p.add_argument('--skip', type=int, default=3, help='process heavy inference every N frames')
    p.add_argument('--gallery_fp', default='gallery.pkl', help='path to persist gallery')
    args = p.parse_args()
    run(args.checkpoint, args.gallery, args.config, device=args.device, threshold=args.threshold, 
        cam_id=args.cam, gallery_fp=args.gallery_fp, use_onnx=args.onnx, frame_skip=args.skip)