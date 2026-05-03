import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class EmbeddingNet(nn.Module):
    """Backbone -> embedding (L2 normalized)
    Uses torchvision ResNet50 by default (pretrained=True)."""

    def __init__(self, embedding_size=512, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet50(weights=weights)
            in_feat = net.fc.in_features
            # remove fc
            modules = list(net.children())[:-1]
            self.backbone = nn.Sequential(*modules)
        elif backbone == 'mobilenet_v2':
            weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            net = tv_models.mobilenet_v2(weights=weights)
            in_feat = net.classifier[1].in_features
            modules = list(net.features)
            self.backbone = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1,1)))
        else:
            raise ValueError('Unsupported backbone')

        self.embedding = nn.Linear(in_feat, embedding_size)
        # initialize embedding layer
        try:
            nn.init.xavier_normal_(self.embedding.weight)
            if self.embedding.bias is not None:
                nn.init.constant_(self.embedding.bias, 0.0)
        except Exception:
            pass

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def export_onnx(self, file_path, input_size=(1, 3, 112, 112), device='cpu'):
        """Xuất mô hình sang định dạng ONNX để tăng tốc độ chạy trên CPU."""
        self.to(device)
        self.eval()
        dummy_input = torch.randn(*input_size).to(device)
        torch.onnx.export(self, dummy_input, file_path, 
                         input_names=['input'], output_names=['embedding'],
                         dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
                         opset_version=12)
        print(f"Đã xuất mô hình ONNX thành công tại: {file_path}")