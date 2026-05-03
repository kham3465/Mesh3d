# coding: utf-8

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

try:
    from .nms.cpu_nms import cpu_nms
except ImportError:
    cpu_nms = None
from .nms.py_cpu_nms import py_cpu_nms


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    
    if cpu_nms is not None:
        try:
            # Ép kiểu về float32 để tránh lỗi mismatch buffer
            return cpu_nms(dets.astype('float32'), thresh)
        except Exception:
            # Nếu vẫn lỗi (do kiến trúc CPU/Windows), dùng bản Python thuần
            pass
    return py_cpu_nms(dets, thresh)
