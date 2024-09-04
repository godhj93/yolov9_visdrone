import torch

from utils.general import check_version

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# QAT compatibility

# def make_anchors_qat(feats, strides, grid_cell_offset=0.5):
#     """Generate anchors from features."""
#     anchor_points, stride_tensor = [], []
#     assert feats is not None
#     dtype, device = feats[0].dtype, feats[0].device
#     for i, stride in enumerate(strides):
#         _, _, h, w = feats[i].shape
#         sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
#         sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
#         sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
#         anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
#         stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
#     return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    '''
    x1y1 = anchor_points - lt 이게 문제임, torch.split은 문제가 없음
    x1y1은 정수형태임, 양자화 시 문제가 발생할 여지가 있음
    x2y2도 마찬가지임
    '''
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    # if xywh: # True 이므로 생략가능
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat((c_xy, wh), dim)  # xywh bbox

    # return torch.cat((x1y1, x2y2), dim)  # xyxy bbox # 사용하지 않음


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

