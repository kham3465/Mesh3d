import torch
import os
import sys

# Thêm thư mục gốc vào sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from recognition.models import EmbeddingNet

def convert():
    checkpoint_path = os.path.join('results', 'checkpoints', 'latest_recog_ck.pth')
    onnx_path = os.path.join('results', 'checkpoints', 'latest_recog_ck.onnx')
    
    if not os.path.exists(checkpoint_path):
        print(f" Không tìm thấy file {checkpoint_path}")
        return

    model = EmbeddingNet(backbone='resnet50', pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt))
    model.export_onnx(onnx_path)

if __name__ == '__main__':
    convert()