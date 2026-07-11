# api_server.py
import onnxruntime as ort
import sys
import os
import cv2
import yaml
import torch
import numpy as np
import base64
import io
import uuid
from PIL import Image
import boto3
from typing import List, Optional

# FastAPI và Pydantic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

# Import các module của dự án
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
from utils.functions import crop_img
from recognition.dataset import default_transforms
from db_utils import enroll_person_db, recognize_person_db, delete_person_db, list_people_db, update_person_name_db, update_person_image_url_db

# --- 1. Khởi tạo các model và cấu hình một lần duy nhất khi server khởi động ---
print("Khởi tạo server và tải các model...")

# Tải cấu hình TDDFA
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Khởi tạo các model ONNX để có hiệu năng cao nhất trên CPU
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)

# Tải model nhận diện
recog_checkpoint_path = 'results/checkpoints/latest_recog_ck.onnx'
recog_session = None
if os.path.exists(recog_checkpoint_path):
    print(f"Tải model nhận diện ONNX từ {recog_checkpoint_path}...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    recog_session = ort.InferenceSession(recog_checkpoint_path, sess_options, providers=['CPUExecutionProvider'])
else:
    print(f"CẢNH BÁO: Không tìm thấy model nhận diện tại {recog_checkpoint_path}. Các chức năng sẽ bị vô hiệu hóa.")

recog_transform = default_transforms(112)
threshold = 0.75 # Sử dụng ngưỡng đã được tinh chỉnh

print("Server đã sẵn sàng nhận yêu cầu.")
# --- Kết thúc phần khởi tạo ---

# --- Cấu hình S3 ---
S3_BUCKET = 'iot-project'
S3_REGION = 'ap-southeast-1'
s3_client = boto3.client('s3', region_name=S3_REGION)


# -----------------------



# --- 2. Định nghĩa các model dữ liệu cho request và response ---
class EnrollResponse(BaseModel):
    id: int
    message: str

class EnrollRequest(BaseModel):
    images: List[str] = Field(..., description="Danh sách các ảnh đã được mã hóa Base64.")
    name: Optional[str] = Field(None, description="Tên người dùng (tùy chọn).")

class RecognizeRequest(BaseModel):
    images: List[str] = Field(..., description="Danh sách các ảnh đã được mã hóa Base64.")

class UpdateNameRequest(BaseModel):
    name: str = Field(..., description="Tên mới cần cập nhật cho người dùng.")

class Identity(BaseModel):
    id: int
    name: str
    image_url: Optional[str] = None

class RecognizeResponse(BaseModel):
    identity: str
    confidence: float
    message: str

class DeleteResponse(BaseModel):
    status: str
    message: str
# --- 3. Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="3DDFA Face Recognition API",
    description="API cho việc đăng ký và nhận diện khuôn mặt sử dụng 3DDFA_V2.",
    version="1.0.0"
)
# Tạo thư mục static nếu chưa có và mount nó
STATIC_DIR = "static/enroll_images"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 4. Các hàm xử lý logic ---
def process_base64_images(base64_images: List[str]):
    """Hàm chung để xử lý một loạt ảnh base64 và trích xuất embedding."""
    collected_embs = []
    first_image_bytes = None
    if not base64_images or recog_session is None:
        return collected_embs, first_image_bytes

    for b64_string in base64_images:
        try:
            # Giải mã base64 thành bytes
            img_bytes = base64.b64decode(b64_string)
            if first_image_bytes is None:
                first_image_bytes = img_bytes
            # Đọc ảnh từ bytes
            img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except Exception:
            continue # Bỏ qua nếu ảnh không hợp lệ

        boxes = face_boxes(img_bgr)
        if not boxes:
            continue

        boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[:1]
        param_lst, roi_box_lst = tddfa(img_bgr, boxes)
        if not param_lst:
            continue

        face_crop = crop_img(img_bgr, roi_box_lst[0])
        if face_crop.size == 0:
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        input_tensor = recog_transform(pil_crop).unsqueeze(0).numpy()

        inputs = {recog_session.get_inputs()[0].name: input_tensor}
        emb = torch.from_numpy(recog_session.run(None, inputs)[0]).squeeze(0)
        collected_embs.append(emb)

    return collected_embs, first_image_bytes

# --- 5. Định nghĩa các API Endpoints ---
@app.post("/enroll", response_model=EnrollResponse, summary="Đăng ký người dùng mới")
async def enroll(request: EnrollRequest):
    """
    Nhận một loạt ảnh (tối đa 10) để đăng ký người dùng mới.
    Hệ thống sẽ tự động tạo một ID duy nhất, dùng ID đó làm tên,
    tải ảnh đầu tiên lên S3 và trả về ID.
    """
    if not request.images:
        raise HTTPException(status_code=400, detail="Vui lòng gửi ít nhất một ảnh để đăng ký.")
    if len(request.images) > 10:
        raise HTTPException(status_code=400, detail="Vui lòng gửi tối đa 10 ảnh để đăng ký.")

    collected_embs, first_image_bytes = process_base64_images(request.images)
    if not collected_embs:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt hợp lệ trong các ảnh đã gửi.")

    mean_emb = torch.stack(collected_embs, dim=0).mean(0)
    mean_emb_np = mean_emb.cpu().numpy()

    # Bước 1: Đăng ký embedding và lấy ID mới, image_url tạm thời là None
    new_id, message = enroll_person_db(mean_emb_np, None)
    if new_id == -1:
        raise HTTPException(status_code=500, detail=message)

    # Gán tên cho identity vừa tạo (nếu client gửi kèm)
    if request.name:
        name_message = update_person_name_db(new_id, request.name)
        if "Lỗi" in name_message or "Không tìm thấy" in name_message:
            # Trùng tên (cột name unique) -> 409, identity không tên đã được tạo (chấp nhận, không rollback)
            status_code = 409 if "đã tồn tại" in name_message else 500
            raise HTTPException(status_code=status_code, detail=name_message)

    # Bước 2: Tải ảnh đầu tiên lên S3 (nếu có) và cập nhật lại DB
    if first_image_bytes:
        try:
            # Tạo tên file duy nhất
            image_filename = f"{uuid.uuid4()}.jpg"
            # Tạo key S3 theo cấu trúc documents/{userID}/{imageID}.jpg
            s3_key = f"documents/{new_id}/{image_filename}"
            # Tải file lên S3 với quyền đọc công khai
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=first_image_bytes, ContentType='image/jpeg')

            # Tạo URL để lưu vào DB
            image_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
            # Cập nhật lại image_url cho người dùng vừa tạo
            update_person_image_url_db(new_id, image_url)
        except Exception as e:
            # Nếu tải S3 lỗi, vẫn trả về thành công nhưng có cảnh báo
            message += f" (Cảnh báo: Không thể tải ảnh lên S3: {e})"

    return EnrollResponse(id=new_id, message=message)

@app.post("/recognize", response_model=RecognizeResponse, summary="Nhận diện người dùng")
def recognize(request: RecognizeRequest):
    """
    Nhận một loạt ảnh (tối đa 5) và trả về danh tính của người trong ảnh.
    """
    if len(request.images) > 5:
        raise HTTPException(status_code=400, detail="Vui lòng gửi tối đa 5 ảnh để nhận diện.")

    collected_embs, _ = process_base64_images(request.images)
    if not collected_embs:
        return RecognizeResponse(identity="Unknown", confidence=0.0, message="Không thể xử lý khuôn mặt nào để nhận diện.")

    mean_emb = torch.stack(collected_embs, dim=0).mean(0)
    mean_emb_np = mean_emb.cpu().numpy()

    best_label, best_sim = recognize_person_db(mean_emb_np, threshold=threshold)
    message = f"Nhận diện dựa trên {len(collected_embs)} ảnh hợp lệ."
    return RecognizeResponse(identity=best_label, confidence=best_sim, message=message)

@app.get("/", summary="Kiểm tra trạng thái API")
def read_root():
    return {"status": "API is running"}

@app.get("/identities", response_model=List[Identity], summary="Lấy danh sách tất cả người dùng đã đăng ký")
def get_identities():
    """
    Trả về một danh sách các đối tượng, mỗi đối tượng chứa `id`, `name`, và `image_url` của người dùng.
    """
    return list_people_db()

@app.put("/identities/{id}", response_model=DeleteResponse, summary="Cập nhật tên người dùng")
def update_identity_name(id: int, request: UpdateNameRequest):
    """Cập nhật tên cho một người dùng dựa trên ID của họ."""
    message = update_person_name_db(id, request.name)
    if "Lỗi" in message or "Không tìm thấy" in message:
        # Lỗi trùng tên (409 Conflict) hoặc không tìm thấy (404 Not Found)
        status_code = 409 if "đã tồn tại" in message else 404
        raise HTTPException(status_code=status_code, detail=message)
    return DeleteResponse(status="success", message=message)

@app.delete("/identities/{id}", response_model=DeleteResponse, summary="Xóa một người dùng")
def delete_identity(id: int):
    """Xóa một người dùng khỏi database dựa trên ID."""
    message = delete_person_db(id)
    if "Lỗi" in message or "Không tìm thấy" in message:
        raise HTTPException(status_code=404, detail=message)
    return DeleteResponse(status="success", message=message)