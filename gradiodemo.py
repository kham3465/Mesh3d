# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
import sys
import os

import cv2
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import crop_img
from recognition.dataset import default_transforms
from db_utils import enroll_person_db, recognize_person_db, delete_person_db, list_people_db # THAY ĐỔI: Import từ db_utils
import onnxruntime as ort
import torch

from PIL import Image
import numpy as np

import gradio as gr

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)

# --- Thêm phần nhận diện ---
recog_checkpoint_path = 'results/checkpoints/latest_recog_ck.onnx'
device = torch.device('cpu')
threshold = 0.75 # Tăng ngưỡng để hệ thống "khó tính" hơn
recog_transform = default_transforms(112)
recog_session = None

# Tải mô hình nhận diện ONNX
if os.path.exists(recog_checkpoint_path):
    print(f"Loading ONNX recognition model from {recog_checkpoint_path}...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    recog_session = ort.InferenceSession(recog_checkpoint_path, sess_options, providers=['CPUExecutionProvider'])
else:
    print(f"Recognition model not found at {recog_checkpoint_path}. Recognition will be disabled.")

# --- Kết thúc phần nhận diện ---

def process_images_for_embedding(imgs):
    """Trích xuất embedding từ một loạt ảnh, trả về list các embedding."""
    if not imgs or recog_session is None:
        return [], None

    collected_embs = []
    first_img_processed = None

    # `imgs` bây giờ là một danh sách các đường dẫn file tạm thời
    for i, img_path in enumerate(imgs):
        # Đọc ảnh từ đường dẫn file
        img_np = cv2.imread(img_path.name) # .name chứa đường dẫn thực sự
        if img_np is None:
            continue

        # face detection
        boxes = face_boxes(img_np)
        if not boxes:
            continue

        # Chỉ xử lý khuôn mặt lớn nhất trong mỗi ảnh
        boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[:1]
        
        param_lst, roi_box_lst = tddfa(img_np, boxes)
        if not param_lst:
            continue

        roi_box = roi_box_lst[0]
        face_crop = crop_img(img_np, roi_box)
        if face_crop.size == 0:
            continue
        
        pil_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        input_tensor = recog_transform(pil_crop).unsqueeze(0).numpy()

        inputs = {recog_session.get_inputs()[0].name: input_tensor}
        emb = torch.from_numpy(recog_session.run(None, inputs)[0]).squeeze(0)
        collected_embs.append(emb)

        # Vẽ lưới 3D lên ảnh đầu tiên để làm ảnh output
        if i == 0:
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            first_img_processed = render(img_np, ver_lst, tddfa.tri, alpha=0.6, show_flag=False)

    if first_img_processed is None:
        if imgs:
            # Nếu không xử lý được ảnh nào, trả về ảnh gốc đầu tiên
            first_img_processed = cv2.imread(imgs[0].name)
        else:
            first_img_processed = None

    return collected_embs, first_img_processed

def recognize_person(imgs):
    """API nhận diện: nhận lô ảnh, trả về danh tính."""
    if not imgs:
        return None, "Vui lòng tải lên tối đa 5 ảnh."
    if len(imgs) > 5:
        return None, "Vui lòng tải lên tối đa 5 ảnh để nhận diện."

    collected_embs, first_img_processed = process_images_for_embedding(imgs)

    if not collected_embs or first_img_processed is None:
        return first_img_processed, "Không thể xử lý khuôn mặt nào để nhận diện."

    # Tính trung bình các embedding
    mean_emb = torch.stack(collected_embs, dim=0).mean(0)
    mean_emb_np = mean_emb.cpu().numpy()

    # THAY ĐỔI: Gọi hàm nhận diện từ DB
    best_label, best_sim = recognize_person_db(mean_emb_np, threshold=threshold)
    # Đọc dữ liệu byte của ảnh đầu tiên để lưu vào log
    first_image_bytes = None
    if imgs:
        try:
            with open(imgs[0].name, 'rb') as f:
                first_image_bytes = f.read()
        except Exception:
            pass # Bỏ qua nếu không đọc được file
    best_label, best_sim = recognize_person_db(mean_emb_np, threshold=threshold, image_bytes=first_image_bytes)
    
    result_text = f"Danh tính: {best_label}\nĐộ tin cậy: {best_sim:.2f}\n(Dựa trên {len(collected_embs)} ảnh)"
    return first_img_processed, result_text

def enroll_person(imgs, name):
    """API đăng ký: nhận lô ảnh và tên, lưu vào DB."""
    name = name.strip()
    if not name:
        return "Vui lòng nhập tên để đăng ký."
    if not imgs:
        return "Vui lòng tải lên tối đa 10 ảnh để đăng ký."
    if len(imgs) > 10:
        return "Vui lòng tải lên tối đa 10 ảnh để đăng ký."

    collected_embs, _ = process_images_for_embedding(imgs)

    if not collected_embs:
        return "Đăng ký thất bại: Không tìm thấy khuôn mặt hợp lệ trong các ảnh đã tải lên."

    # Tính trung bình embedding
    mean_emb = torch.stack(collected_embs, dim=0).mean(0)
    mean_emb_np = mean_emb.cpu().numpy()
    
    # THAY ĐỔI: Gọi hàm đăng ký vào DB
    message = enroll_person_db(name, mean_emb_np)
    return message

def delete_person_ui(name_to_delete):
    """Hàm xử lý cho giao diện xóa người dùng."""
    if not name_to_delete:
        return "Vui lòng chọn một người dùng để xóa.", gr.update() # Không thay đổi dropdown nếu không có gì để xóa
    message = delete_person_db(name_to_delete)
    # Cập nhật lại dropdown sau khi xóa
    return message, gr.update(choices=list_people_db(), value=None)

def refresh_dropdown():
    """Hàm để làm mới danh sách người dùng trong dropdown."""
    return gr.update(choices=list_people_db())

title = "3DDFA V2 - Hệ thống sinh trắc học"
description = "Sử dụng các tab bên dưới cho các chức năng khác nhau. Hệ thống sử dụng Database để lưu trữ."
article = "<p style='text-align: center'><a href='https://github.com/cleardusk/3DDFA_V2'>Github Repo</a></p>"

with gr.Blocks() as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)

    with gr.Tabs():
        with gr.TabItem("Nhận diện (Tối đa 5 ảnh)"):
            with gr.Row():
                with gr.Column():
                    rec_input_images = gr.File(label="Tải lên các ảnh cần nhận diện", file_count="multiple", file_types=["image"], type="filepath")
                    rec_button = gr.Button("Nhận diện", variant="primary")
                with gr.Column():
                    rec_output_image = gr.Image(label="Kết quả 3D", type="numpy")
                    rec_output_text = gr.Textbox(label="Kết quả nhận diện")
            rec_button.click(recognize_person, inputs=[rec_input_images], outputs=[rec_output_image, rec_output_text])

        with gr.TabItem("Đăng ký (Tối đa 10 ảnh)"):
            with gr.Row():
                with gr.Column():
                    enr_input_name = gr.Textbox(label="Nhập tên/ID người đăng ký")
                    enr_input_images = gr.File(label="Tải lên nhiều ảnh (tối đa 10) với các góc mặt khác nhau để tăng độ chính xác.", file_count="multiple", file_types=["image"], type="filepath")
                    enr_button = gr.Button("Đăng ký", variant="primary")
                with gr.Column():
                    enr_output_text = gr.Textbox(label="Trạng thái đăng ký")
            enr_button.click(enroll_person, inputs=[enr_input_images, enr_input_name], outputs=[enr_output_text])

        with gr.TabItem("Quản lý người dùng"):
            with gr.Row():
                with gr.Column(scale=3):
                    delete_dropdown = gr.Dropdown(label="Chọn người dùng để xóa", choices=list_people_db())
                    delete_button = gr.Button("Xóa người dùng đã chọn", variant="stop")
                with gr.Column(scale=1):
                    refresh_button = gr.Button("Làm mới danh sách")
            with gr.Row():
                delete_status_text = gr.Textbox(label="Trạng thái xóa")
            
            delete_button.click(delete_person_ui, inputs=[delete_dropdown], outputs=[delete_status_text, delete_dropdown])
            refresh_button.click(refresh_dropdown, inputs=None, outputs=[delete_dropdown])

    
    gr.Markdown(article)

demo.launch(share=True, server_name="0.0.0.0", theme=gr.themes.Soft())
