from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from projects.mmdet3d_plugin.core.box3d import *


def interp_anchor_to_traj(spatial_anchor, dist_anchor):
    """
    spatial_anchor: [N, P, 2]   float32  固定距离间隔的 anchor 路点
    dist_anchor:    [M, K]      float32  想要查询的累计距离
    return traj:    [N, M, K, 2]          插值后的轨迹
    """
    spatial_anchor = torch.cat([spatial_anchor[:, :1]*0, spatial_anchor], dim=1)
    N, P, _ = spatial_anchor.shape
    M, K = dist_anchor.shape

    # 1. 计算 anchor 的逐段长度并累加，得到 s_anchor [N, P]
    seg = torch.diff(spatial_anchor, dim=1)               # [N, P-1, 2]
    seg_len = seg.norm(dim=-1)                            # [N, P-1]
    s_anchor = F.pad(seg_len.cumsum(dim=1), (1, 0))       # [N, P]  第 0 点为 0

    # 2. 把 dist_anchor 扩到 [N, M, K] 方便并行查询
    query = dist_anchor.unsqueeze(0).expand(N, -1, -1)    # [N, M, K]
    # query_clip = query.clamp(max=s_anchor[:, None, -1:])
    query_clip = query

    # 3. 找到每一段落在哪个区间
    #    searchsorted 需要 query 最后一维是 K，所以把 N,M 合并，做完再 reshape 回来
    query_flat = query_clip.reshape(N, -1)                     # [N, M*K]
    idx = torch.searchsorted(s_anchor, query_flat, right=True) - 1  # [N, M*K]
    idx = idx.clamp(min=0, max=P-2)                       # 保证不越界

    # 4. 取区间端点
    s0 = torch.gather(s_anchor, 1, idx)                   # [N, M*K]
    s1 = torch.gather(s_anchor, 1, idx+1)
    xy0 = torch.gather(spatial_anchor, 1,
                    idx.unsqueeze(-1).expand(-1, -1, 2))  # [N, M*K, 2]
    xy1 = torch.gather(spatial_anchor, 1,
                    (idx+1).unsqueeze(-1).expand(-1, -1, 2))

    # 5. 线性插值
    w = (query_flat - s0) / (s1 - s0 + 1e-6)              # [N, M*K]
    xy_flat = xy0 + w.unsqueeze(-1) * (xy1 - xy0)         # [N, M*K, 2]

    # 6. reshape 回目标形状
    traj = xy_flat.reshape(N, M, K, 2)
    mask = (query < s_anchor[:, None, -1:]).float()
    return traj, mask

def get_col_mask(
    traj,
    traj_mask,
    det_output,
    motion_output,
    score_thresh,
    num_motion_mode=1,
):
    def cat_with_zero(traj):
        zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
        traj_cat = torch.cat([zeros, traj], dim=-2)
        return traj_cat

    def cat_with_ones(traj_mask):
        ones = traj_mask.new_ones(traj_mask.shape[:-1] + (1,))
        traj_mask_cat = torch.cat([ones, traj_mask], dim=-1)
        return traj_mask_cat

    def get_yaw(traj, start_yaw=np.pi/2, static_dis_thresh=0.5):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        yaw[..., 0] = start_yaw
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)

    det_classification = det_output["classification"][-1].sigmoid()
    det_anchors = det_output["prediction"][-1]
    det_confidence = det_classification.max(dim=-1).values
    motion_cls = motion_output["classification"][-1].sigmoid()
    motion_reg = motion_output["prediction"][-1]

    ## ego
    traj_cat = cat_with_zero(traj).flatten(0,1)
    traj_mask = cat_with_ones(traj_mask).flatten(0, 1).bool()
    traj_cat = torch.where(
        traj_mask.unsqueeze(-1),
        traj_cat,
        traj_cat.new_tensor([1e4]),
    )
    lat_mode, lon_mode, num_pts = traj.shape[:3]
    ego_box = det_anchors.new_zeros(lat_mode * lon_mode, num_pts + 1, 7)
    ego_box[..., [X, Y]] = traj_cat
    ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) #* dim_scale
    ego_box[..., [YAW]] = get_yaw(traj_cat)
    ego_box = ego_box[..., 1:, :]
    ego_corner = get_corners(ego_box)

    ## motion
    motion_reg = motion_reg[..., :num_pts, :].cumsum(-2)
    motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
    _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
    motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, num_pts + 1, 2)
    motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

    motion_boxs = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
    motion_boxs[..., [X, Y]] = motion_reg
    motion_boxs[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
    box_yaw = torch.atan2(
        det_anchors[..., SIN_YAW],
        det_anchors[..., COS_YAW],
    )
    motion_boxs[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))
    motion_boxs = motion_boxs[..., 1:, :]

    batch_size = motion_boxs.shape[0]
    col_masks = []
    for bs in range(batch_size):
        score_mask = det_confidence[bs] > score_thresh
        motion_box = motion_boxs[bs, score_mask].flatten(0, 1)
        motion_corner = get_corners(motion_box)
        col = sat_2d_cuda(ego_corner, motion_corner)
        # visualize_boxes_single(ego_corner, motion_corner, col)
        col = col.unflatten(0, (lat_mode, lon_mode))
        col_masks.append(col)
    return col_masks

def get_corners(box):
    """
    box: (..., 7)  (x, y, z, w, l, h, yaw)
    return: (..., 4, 2)  四个角点
    """
    x, y, _, w, l, _, yaw = [box[..., i] for i in range(7)]
    cos, sin = torch.cos(yaw), torch.sin(yaw)

    # 局部坐标系下的 4 个角
    dx = torch.stack([ w/2,  w/2, -w/2, -w/2], dim=-1)
    dy = torch.stack([ l/2, -l/2, -l/2,  l/2], dim=-1)

    # 旋转到全局
    rx = dx * cos[..., None] - dy * sin[..., None]
    ry = dx * sin[..., None] + dy * cos[..., None]

    return torch.stack([x[..., None] + rx,
                    y[..., None] + ry], dim=-1)     # (..., 4, 2)

def sat_2d_cuda(corners_a: torch.Tensor,
                corners_b: torch.Tensor,
                aabb_thresh: float = 0.0):
    """
    2-D SAT 碰撞检测（纯 PyTorch + CUDA）
    corners_a : (A, T, 4, 2)  float32/float16
    corners_b : (B, T, 4, 2)
    aabb_thresh : 允许 AABB 外扩一点，默认 0
    return    : (A, B, T) bool  True=碰撞
    """
    device = corners_a.device
    dtype  = corners_a.dtype
    A, T, _, _ = corners_a.shape
    B = corners_b.shape[0]

    # ---------- 1. AABB 预剪枝 ----------
    # (A,T,2)  (minx/miny)
    a_min = corners_a.min(dim=2)[0]          # (A,T,2)
    a_max = corners_a.max(dim=2)[0]
    b_min = corners_b.min(dim=2)[0]          # (B,T,2)
    b_max = corners_b.max(dim=2)[0]

    # 外扩
    if aabb_thresh > 0:
        a_min -= aabb_thresh;  a_max += aabb_thresh
        b_min -= aabb_thresh;  b_max += aabb_thresh

    # 广播比较  (A,B,T)
    no_overlap = (a_max[:, None, :, 0] < b_min[None, :, :, 0]) | \
                (b_max[None, :, :, 0] < a_min[:, None, :, 0]) | \
                (a_max[:, None, :, 1] < b_min[None, :, :, 1]) | \
                (b_max[None, :, :, 1] < a_min[:, None, :, 1])
    valid_mask = ~no_overlap                 # (A,B,T)
    # 如果全 False 可直接返回
    if valid_mask.sum() == 0:
        return torch.zeros((A, B, T), dtype=torch.bool, device=device)

    # 只保留需要 SAT 的 (a,b,t) 三元组 → 稀疏列表
    a_idx, b_idx, t_idx = torch.where(valid_mask)   # 1-D 长 ~N
    corners_a_sparse = corners_a[a_idx, t_idx]      # (N,4,2)
    corners_b_sparse = corners_b[b_idx, t_idx]      # (N,4,2)

    # ---------- 2. 计算 8 条边 ----------
    # (N,4,2)
    edges_a = corners_a_sparse[:, [0,1,2,3]] - corners_a_sparse[:, [1,2,3,0]]
    edges_b = corners_b_sparse[:, [0,1,2,3]] - corners_b_sparse[:, [1,2,3,0]]
    edges   = torch.cat([edges_a, edges_b], dim=1)  # (N,8,2)

    # ---------- 3. 法向量并归一化 ----------
    axes = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)  # (N,8,2)
    axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)

    # ---------- 4. 投影 ----------
    # (N,4,8)
    proj_a = torch.einsum('npi,nji->npj', corners_a_sparse, axes)
    proj_b = torch.einsum('npi,nji->npj', corners_b_sparse, axes)
    min_a, max_a = proj_a.min(dim=1)[0], proj_a.max(dim=1)[0]  # (N,8)
    min_b, max_b = proj_b.min(dim=1)[0], proj_b.max(dim=1)[0]

    # ---------- 5. 重叠判断 ----------
    overlap = (max_a >= min_b) & (max_b >= min_a)        # (N,8)
    collide = overlap.all(dim=1)                         # (N,)

    # ---------- 6. 填回稠密张量 ----------
    out = torch.zeros((A, B, T), dtype=torch.bool, device=device)
    out[a_idx, b_idx, t_idx] = collide
    return out

def visualize_boxes_single(
    ego_corners,      # [A, T, 4, 2]
    motion_corners,   # [B, T, 4, 2]
    col_mask,         # [A, B, T]  bool
    index=None,
    px_per_m=40,
    margin=1.0,
    thickness=4,
    alpha_early=0.25,   # 最早一帧透明度
    alpha_late=0.95):   # 最后一帧透明度
    """
    返回一张 uint8 BGR 图像，所有时刻画在一起
    """
    device = ego_corners.device
    A, T, _, _ = ego_corners.shape
    B = motion_corners.shape[0]

    # 1. 画布大小
    all_pts = torch.cat([ego_corners.view(-1, 2),
                        motion_corners.view(-1, 2)], 0)
    x_min, y_min = -30.0, -30.0
    x_max, y_max =  30.0,  30.0
    W_m, H_m = x_max - x_min, y_max - y_min
    W_px, H_px = int(W_m * px_per_m), int(H_m * px_per_m)

    # 2. 坐标 → 像素
    def to_px(pt):
        x, y = pt[..., 0], pt[..., 1]
        u = ((x - x_min) * px_per_m).long()
        v = ((y_max - y) * px_per_m).long()
        return torch.stack([u, v], -1)

    ego_px  = to_px(ego_corners)    # [A, T, 4, 2]
    motion_px = to_px(motion_corners)

    # 3. 准备一张透明底板
    overlay = np.zeros((H_px, W_px, 4), dtype=np.float32)  # BGR-A

    # 4. 时间维度透明度线性插值
    alphas = np.linspace(alpha_early, alpha_late, T)

    # 6. 再画 ego（红/蓝）
    for t in range(T):
        alpha = alphas[t]
        for a in range(A):
            has_col = col_mask[a, :, t].any().item()
            color_bgr = (0, 0, 255) if has_col else (255, 0, 0)
            # 把 BGR 转成 BGR-A
            color_bgra = (*color_bgr, alpha*255)
            pts = ego_px[a, t].cpu().numpy()
            cv2.polylines(overlay, [pts], isClosed=True,
                        color=color_bgra, thickness=thickness)

    # 5. 先画所有 motion（绿色，早淡晚深）
    for t in range(T):
        alpha = alphas[t]
        for b in range(B):
            pts = motion_px[b, t].cpu().numpy()
            cv2.polylines(overlay, [pts], isClosed=True,
                        color=(0, 255, 0, alpha*255),
                        thickness=thickness)

    # 7. 把带透明通道的 overlay 叠到白色背景
    bgr   = (overlay[:, :, :3]).astype(np.uint8)
    alpha = overlay[:, :, 3:4] / 255.0
    white = np.ones_like(bgr, dtype=np.uint8) * 255
    img = cv2.convertScaleAbs(bgr * alpha + white * (1 - alpha))
    if index is None:
        if not hasattr(visualize_boxes_single, 'index'):
            visualize_boxes_single.index = 0
        else:
            visualize_boxes_single.index += 1
        index = visualize_boxes_single.index

    cv2.imwrite(f'all_T_in_one_{str(index).zfill(3)}.jpg', img)
    return img


def get_col_label(
    traj,
    traj_mask,
    data,
    num_motion_mode=1,
):
    def cat_with_zero(traj):
        zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
        traj_cat = torch.cat([zeros, traj], dim=-2)
        return traj_cat

    def cat_with_ones(traj_mask):
        ones = traj_mask.new_ones(traj_mask.shape[:-1] + (1,))
        traj_mask_cat = torch.cat([ones, traj_mask], dim=-1)
        return traj_mask_cat

    def get_yaw(traj, start_yaw=np.pi/2, static_dis_thresh=0.5):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        yaw[..., 0] = start_yaw
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)

    ## ego
    debug = 0
    if debug:
        traj = traj[::20, 2:3]
        traj_mask = traj_mask[::20, 2:3]

    traj_cat = cat_with_zero(traj).flatten(0,1)
    traj_mask = cat_with_ones(traj_mask).flatten(0, 1).bool()
    traj_cat = torch.where(
        traj_mask.unsqueeze(-1),
        traj_cat,
        traj_cat.new_tensor([1e4]),
    )
    lat_mode, lon_mode, num_pts = traj.shape[:3]
    ego_box = traj.new_zeros(lat_mode * lon_mode, num_pts + 1, 7)
    ego_box[..., [X, Y]] = traj_cat
    ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) #* dim_scale
    ego_box[..., [YAW]] = get_yaw(traj_cat)
    ego_box = ego_box[..., 1:, :]
    ego_corner = get_corners(ego_box)

    ## motion
    boxes = data["gt_bboxes_3d"]
    motions = data["gt_agent_fut_trajs"]
    motion_masks = data["gt_agent_fut_masks"]
    bs = len(boxes)
    max_len = max([box.shape[0] for box in boxes])

    motion_boxs = traj.new_zeros([bs, max_len, num_pts + 1, 7])
    motion_boxs[..., :2] = -1e4
    for b in range(bs):
        box = boxes[b]
        motion = motions[b]
        motion_mask = motion_masks[b]
        motion = cat_with_zero(motion)
        motion_mask = cat_with_ones(motion_mask)
        num_box = len(box)

        motion = box[:, :2].unsqueeze(1) + motion.cumsum(dim=-2)
        motion = torch.where(
            motion_mask.unsqueeze(-1).bool(),
            motion,
            motion.new_tensor([-1e4]),
        )
        motion_boxs[b, :num_box, :, [X, Y]] = motion[:, :num_pts + 1]
        motion_boxs[b, :num_box, :, [W, L, H]] = box[:, [W, L, H]].unsqueeze(1)
        motion_boxs[b, :num_box, :, [YAW]] = box[:, [YAW]].unsqueeze(1)

    motion_boxs = motion_boxs[..., 1:, :]

    batch_size = motion_boxs.shape[0]
    col_masks = []
    for bs in range(batch_size):
        motion_box = motion_boxs[bs]
        motion_corner = get_corners(motion_box)
        col = sat_2d_cuda(ego_corner, motion_corner)
        if debug:
            visualize_boxes_single(ego_corner, motion_corner, col, index=data["index"][bs].item())
        col = col.flatten(1, 2).any(dim=-1).unflatten(0, (lat_mode, lon_mode))
        col_masks.append(col)
    return col_masks



def get_col_label_bs(
    traj,
    traj_mask,
    data,
    num_motion_mode=1,
    num_pts=6,
):
    num_pts = traj.shape[2]
    def cat_with_zero(traj):
        zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
        traj_cat = torch.cat([zeros, traj], dim=-2)
        return traj_cat

    def cat_with_ones(traj_mask):
        ones = traj_mask.new_ones(traj_mask.shape[:-1] + (1,))
        traj_mask_cat = torch.cat([ones, traj_mask], dim=-1)
        return traj_mask_cat

    def get_yaw(traj, start_yaw=np.pi/2, static_dis_thresh=0.5):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        yaw[..., 0] = start_yaw
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)

    ## ego
    debug = 0
    if debug:
        traj = traj[::20, 2:3]
        traj_mask = traj_mask[::20, 2:3]

    def get_ego_corner(traj, traj_mask):
        traj_cat = cat_with_zero(traj)
        traj_mask = cat_with_ones(traj_mask).bool()
        traj_cat = torch.where(
            traj_mask.unsqueeze(-1),
            traj_cat,
            traj_cat.new_tensor([1e4]),
        )
        num_traj, num_pts = traj.shape[:2]
        ego_box = traj.new_zeros(num_traj, num_pts + 1, 7)
        ego_box[..., [X, Y]] = traj_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) #* dim_scale
        ego_box[..., [YAW]] = get_yaw(traj_cat)
        ego_box = ego_box[..., 1:, :]
        ego_corner = get_corners(ego_box)
        return ego_corner

    ## motion
    boxes = data["gt_bboxes_3d"]
    motions = data["gt_agent_fut_trajs"]
    motion_masks = data["gt_agent_fut_masks"]
    bs = len(boxes)
    max_len = max([box.shape[0] for box in boxes])

    motion_boxs = traj.new_zeros([bs, max_len, num_pts + 1, 7])
    motion_boxs[..., :2] = -1e4
    for b in range(bs):
        box = boxes[b]
        motion = motions[b]
        motion_mask = motion_masks[b]
        motion = cat_with_zero(motion)
        motion_mask = cat_with_ones(motion_mask)
        num_box = len(box)

        motion = box[:, :2].unsqueeze(1) + motion.cumsum(dim=-2)
        motion = torch.where(
            motion_mask.unsqueeze(-1).bool(),
            motion,
            motion.new_tensor([-1e4]),
        )
        motion_boxs[b, :num_box, :, [X, Y]] = motion[:, :num_pts + 1]
        motion_boxs[b, :num_box, :, [W, L, H]] = box[:, [W, L, H]].unsqueeze(1)
        motion_boxs[b, :num_box, :, [YAW]] = box[:, [YAW]].unsqueeze(1)

    motion_boxs = motion_boxs[..., 1:, :]

    batch_size = motion_boxs.shape[0]
    col_masks = []
    for bs in range(batch_size):
        ego_corner = get_ego_corner(traj[bs], traj_mask[bs])
        motion_box = motion_boxs[bs]
        motion_corner = get_corners(motion_box)
        col = sat_2d_cuda(ego_corner, motion_corner)
        if debug:
            visualize_boxes_single(ego_corner, motion_corner, col, index=data["index"][bs].item())
        col = col.any(dim=1)
        col_masks.append(col)
    col_masks = torch.stack(col_masks, dim=0)
    return col_masks


def interp_anchor_to_traj(spatial_anchor, dist_anchor):
    """
    spatial_anchor: [num_lat_mode, num_lat_fut_ts, 2]   float32  固定距离间隔的 anchor 路点
    dist_anchor:    [num_lon_mode, num_lon_fut_ts]      float32  想要查询的累计距离
    return traj:    [num_lat_mode, num_lon_mode, num_lon_fut_ts, 2]          插值后的轨迹
    """
    spatial_anchor = torch.cat([spatial_anchor[:, :1]*0, spatial_anchor], dim=1)
    N, P, _ = spatial_anchor.shape
    M, K = dist_anchor.shape

    # 1. 计算 anchor 的逐段长度并累加，得到 s_anchor [N, P]
    seg = torch.diff(spatial_anchor, dim=1)               # [N, P-1, 2]
    seg_len = seg.norm(dim=-1)                            # [N, P-1]
    s_anchor = F.pad(seg_len.cumsum(dim=1), (1, 0))       # [N, P]  第 0 点为 0

    # 2. 把 dist_anchor 扩到 [N, M, K] 方便并行查询
    query = dist_anchor.unsqueeze(0).expand(N, -1, -1)    # [N, M, K]
    # query_clip = query.clamp(max=s_anchor[:, None, -1:])
    query_clip = query

    # 3. 找到每一段落在哪个区间
    #    searchsorted 需要 query 最后一维是 K，所以把 N,M 合并，做完再 reshape 回来
    query_flat = query_clip.reshape(N, -1)                     # [N, M*K]
    idx = torch.searchsorted(s_anchor, query_flat, right=True) - 1  # [N, M*K]
    idx = idx.clamp(min=0, max=P-2)                       # 保证不越界

    # 4. 取区间端点
    s0 = torch.gather(s_anchor, 1, idx)                   # [N, M*K]
    s1 = torch.gather(s_anchor, 1, idx+1)
    xy0 = torch.gather(spatial_anchor, 1,
                    idx.unsqueeze(-1).expand(-1, -1, 2))  # [N, M*K, 2]
    xy1 = torch.gather(spatial_anchor, 1,
                    (idx+1).unsqueeze(-1).expand(-1, -1, 2))

    # 5. 线性插值
    w = (query_flat - s0) / (s1 - s0 + 1e-6)              # [N, M*K]
    xy_flat = xy0 + w.unsqueeze(-1) * (xy1 - xy0)         # [N, M*K, 2]

    # 6. reshape 回目标形状
    traj = xy_flat.reshape(N, M, K, 2)
    mask = (query < s_anchor[:, None, -1:]).float()
    return traj, mask

def interp_feature(features, spatial_anchor, dist_anchor):
    """
    features:       [bs, num_lat_mode, num_lat_fut_ts, C]  float32  spatial_anchor 对应的特征（每个path点一个特征）
    spatial_anchor: [num_lat_mode, num_lat_fut_ts, 2]      float32  固定距离间隔的 anchor 路点
    dist_anchor:    [num_lon_mode, num_lon_fut_ts]         float32  想要查询的累计距离（每条speed profile对应的弧长进度）

    return:
      traj_feat: [bs, num_lat_mode, num_lon_mode, num_lon_fut_ts, C]   插值得到的每条轨迹(每个时间点)的特征
      mask:      [bs, num_lat_mode, num_lon_mode, num_lon_fut_ts]      有效标记（query < path总弧长）
    """
    # 对齐你原函数：在最前面补一个0点（位置和特征都补0），保证 s=0 合法
    spatial_anchor0 = torch.cat([spatial_anchor[:, :1] * 0, spatial_anchor], dim=1)   # [N, P+1, 2]
    features0 = torch.cat([features[:, :, :1] * 0, features], dim=2)                  # [bs, N, P+1, C]

    bs, N, P, C = features0.shape  # P = num_lat_fut_ts + 1
    M, K = dist_anchor.shape       # num_lon_mode, num_lon_fut_ts

    # 1) 计算每条path的累计弧长 s_anchor: [N, P]
    seg = torch.diff(spatial_anchor0, dim=1)          # [N, P-1, 2]
    seg_len = seg.norm(dim=-1)                       # [N, P-1]
    s_anchor = F.pad(seg_len.cumsum(dim=1), (1, 0))  # [N, P]，第0点为0

    # 2) query 扩到 [N, M, K]
    query = dist_anchor.unsqueeze(0).expand(N, -1, -1)     # [N, M, K]
    query_flat = query.reshape(N, -1)                      # [N, M*K]

    # 3) searchsorted 找区间 idx: [N, M*K]
    idx = torch.searchsorted(s_anchor, query_flat, right=True) - 1
    idx = idx.clamp(min=0, max=P-2)                        # 防越界

    # 4) 取区间端点的弧长与特征
    s0 = torch.gather(s_anchor, 1, idx)                    # [N, M*K]
    s1 = torch.gather(s_anchor, 1, idx + 1)                # [N, M*K]

    # 在 features0 的 dim=2 (path点维度) 上 gather
    # idx_feat shape 要变成 [bs, N, M*K, C]
    idx_feat0 = idx.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, C)        # [bs, N, M*K, C]
    idx_feat1 = (idx + 1).unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, C)

    f0 = torch.gather(features0, 2, idx_feat0)             # [bs, N, M*K, C]
    f1 = torch.gather(features0, 2, idx_feat1)             # [bs, N, M*K, C]

    # 5) 线性插值（按弧长比例 w）
    w = (query_flat - s0) / (s1 - s0 + 1e-6)               # [N, M*K]
    w = w.unsqueeze(0).unsqueeze(-1)                       # [1, N, M*K, 1] 方便broadcast到bs和C

    f_flat = f0 + w * (f1 - f0)                            # [bs, N, M*K, C]

    # 6) reshape 回目标形状
    traj_feat = f_flat.reshape(bs, N, M, K, C)             # [bs, num_lat_mode, num_lon_mode, num_lon_fut_ts, C]

    # mask：query 小于该path最大弧长为有效
    # s_anchor[:, -1] 是每条path总弧长
    valid = (query < s_anchor[:, None, -1:]).float()       # [N, M, K]
    mask = valid.unsqueeze(0).expand(bs, -1, -1, -1)       # [bs, N, M, K]

    return traj_feat, mask

def interp_feature_lowmem(features, spatial_anchor, dist_anchor, chunk_lon=32):
    """
    Memory-saving version: chunk over num_lon_mode.

    features:       [bs, num_lat_mode, num_lat_fut_ts, C]
    spatial_anchor: [num_lat_mode, num_lat_fut_ts, 2]
    dist_anchor:    [num_lon_mode, num_lon_fut_ts]

    return:
      traj_feat: [bs, num_lat_mode, num_lon_mode, num_lon_fut_ts, C]
      mask:      [bs, num_lat_mode, num_lon_mode, num_lon_fut_ts]
    """
    # 0) prepend zero point like your traj interp
    spatial_anchor0 = torch.cat([spatial_anchor[:, :1] * 0, spatial_anchor], dim=1)  # [N, P+1, 2]
    features0 = torch.cat([features[:, :, :1] * 0, features], dim=2)                  # [bs, N, P+1, C]

    bs, N, P, C = features0.shape              # P = num_lat_fut_ts + 1
    M, K = dist_anchor.shape                   # num_lon_mode, num_lon_fut_ts
    device = features0.device

    # 1) cumulative arc-length per path: s_anchor [N, P]
    seg = torch.diff(spatial_anchor0, dim=1)           # [N, P-1, 2]
    seg_len = seg.norm(dim=-1)                         # [N, P-1]
    s_anchor = F.pad(seg_len.cumsum(dim=1), (1, 0))    # [N, P]

    # 2) flatten features for cheaper gather: [bs*N, P, C]
    feat_flat = features0.reshape(bs * N, P, C)

    # 3) allocate outputs (these are unavoidable if you need full output)
    traj_feat = features0.new_empty((bs, N, M, K, C))
    mask = features0.new_empty((bs, N, M, K))

    # precompute per-path total length for mask
    s_max = s_anchor[:, -1]  # [N]

    # 4) chunk over lon modes
    for m0 in range(0, M, chunk_lon):
        m1 = min(m0 + chunk_lon, M)
        mchunk = m1 - m0

        dist_chunk = dist_anchor[m0:m1]                       # [mchunk, K]
        query = dist_chunk.unsqueeze(0).expand(N, -1, -1)     # [N, mchunk, K]
        query_flat = query.reshape(N, -1)                     # [N, mchunk*K]
        L = query_flat.size(1)                                # L = mchunk*K

        # (optional) clamp query to avoid extrapolation
        # query_flat = torch.minimum(query_flat, s_max[:, None])

        # search interval
        idx = torch.searchsorted(s_anchor, query_flat, right=True) - 1
        idx = idx.clamp(min=0, max=P - 2)                     # [N, L]

        s0 = torch.gather(s_anchor, 1, idx)                   # [N, L]
        s1 = torch.gather(s_anchor, 1, idx + 1)               # [N, L]

        w = (query_flat - s0) / (s1 - s0 + 1e-6)              # [N, L]

        # expand idx for bs via view (no big allocation for idx itself)
        idx0_flat = idx.unsqueeze(0).expand(bs, -1, -1).reshape(bs * N, L)          # [bs*N, L]
        idx1_flat = (idx + 1).unsqueeze(0).expand(bs, -1, -1).reshape(bs * N, L)    # [bs*N, L]

        # gather features at endpoints: [bs*N, L, C]
        f0 = torch.gather(feat_flat, 1, idx0_flat.unsqueeze(-1).expand(-1, -1, C))
        f1 = torch.gather(feat_flat, 1, idx1_flat.unsqueeze(-1).expand(-1, -1, C))

        # interpolate
        w_flat = w.unsqueeze(0).expand(bs, -1, -1).reshape(bs * N, L).unsqueeze(-1) # [bs*N, L, 1]
        f = f0 + w_flat * (f1 - f0)                                                 # [bs*N, L, C]

        # reshape back and write to slice
        f = f.reshape(bs, N, mchunk, K, C)                                          # [bs,N,mchunk,K,C]
        traj_feat[:, :, m0:m1, :, :] = f

        # mask
        valid = (dist_chunk.unsqueeze(0) < s_max[:, None, None]).float()            # [N,mchunk,K]
        mask[:, :, m0:m1, :] = valid.unsqueeze(0).expand(bs, -1, -1, -1)

    return traj_feat, mask
