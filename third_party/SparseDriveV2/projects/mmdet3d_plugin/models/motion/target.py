import torch
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_SAMPLERS

from .rescore_utils import interp_anchor_to_traj, get_col_label, get_col_label_bs

__all__ = ["MotionTarget", "PlanningTarget"]


def get_cls_target(
    reg_preds,
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx

def get_best_reg(
    reg_preds,
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg


@BBOX_SAMPLERS.register_module()
class MotionTarget():
    def __init__(
        self,
    ):
        super(MotionTarget, self).__init__()

    def sample(
        self,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        motion_loss_cache,
    ):
        bs, num_anchor, mode, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        indices = motion_loss_cache['indices']
        num_pos = reg_pred.new_tensor([0])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(gt_reg_target[i]) == 0:
                continue
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx)

        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)
        cls_weight = reg_weight.any(dim=-1)
        best_reg = get_best_reg(reg_pred, reg_target, reg_weight)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


@BBOX_SAMPLERS.register_module()
class PlanningTarget():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_cmd=3,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        if self.num_cmd == 1:
            cmd *= 0

        cls_pred = cls_pred.reshape(bs, self.num_cmd, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.num_cmd, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


@BBOX_SAMPLERS.register_module()
class PlanningTargetV2():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_cmd=3,
    ):
        super(PlanningTargetV2, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        cls_pred = cls_pred[:, 0] # bs, num_mode
        reg_pred = reg_pred[:, 0].cumsum(dim=-2) # bs, num_mode, fut_ts, 2
        gt_reg_target = gt_reg_target.cumsum(dim=-2)
        dist = torch.linalg.norm(gt_reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * gt_reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cls_target = torch.ones([bs, self.ego_fut_mode], dtype=torch.long, device=cls_pred.device)
        cls_target[bs_indices, mode_idx] = 0
        cls_weight = gt_reg_mask.any(dim=-1, keepdim=True) * dist * 100
        cls_weight[bs_indices, mode_idx] = 100

        return cls_pred[..., None], cls_target, cls_weight


@BBOX_SAMPLERS.register_module()
class RefinePlanningTargetV1(): ## cls: one pos; reg: all
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_cmd=3,
    ):
        super(RefinePlanningTargetV1, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        cls_pred = cls_pred[:, 0] # bs, num_mode
        reg_pred = reg_pred[:, 0].cumsum(dim=-2) # bs, num_mode, fut_ts, 2
        gt_reg_target = gt_reg_target.cumsum(dim=-2)
        dist = torch.linalg.norm(gt_reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * gt_reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cls_target = torch.ones([bs, self.ego_fut_mode], dtype=torch.long, device=cls_pred.device)
        cls_target[bs_indices, mode_idx] = 0
        cls_weight = gt_reg_mask.any(dim=-1, keepdim=True) * dist * 100
        cls_weight[bs_indices, mode_idx] = 100

        return cls_pred[..., None], cls_target, cls_weight, reg_pred, gt_reg_target[:, None].repeat(1, self.ego_fut_mode, 1, 1), gt_reg_mask[:, None, :, None].repeat(1, self.ego_fut_mode, 1, 1)


@BBOX_SAMPLERS.register_module()
class RefinePlanningTargetV2(): ### cls: winner takes all; reg: all
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_cmd=3,
        cls_weight=10.0,
    ):
        super(RefinePlanningTargetV2, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.cls_weight = cls_weight

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        num_mode = reg_pred.shape[2]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        if self.num_cmd == 1:
            cmd *= 0

        cls_pred = cls_pred.reshape(bs, self.num_cmd, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.num_cmd, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1) * self.cls_weight
        reg_target = gt_reg_target.unsqueeze(2).repeat(1, 1, num_mode, 1, 1)
        reg_mask = gt_reg_mask.unsqueeze(2).repeat(1, 1, num_mode, 1)

        return cls_pred, cls_target, cls_weight, reg_pred, reg_target, reg_mask


@BBOX_SAMPLERS.register_module()
class RefinePlanningTargetV3(): ### cls: winner taks all; reg: no
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_cmd=3,
        cls_weight=10.0,
    ):
        super(RefinePlanningTargetV3, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.cls_weight = cls_weight

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        num_mode = reg_pred.shape[2]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        if self.num_cmd == 1:
            cmd *= 0

        cls_pred = cls_pred.reshape(bs, self.num_cmd, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.num_cmd, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1) * self.cls_weight
        reg_target = reg_pred
        reg_mask = gt_reg_mask.unsqueeze(2).repeat(1, 1, num_mode, 1) * 0

        return cls_pred, cls_target, cls_weight, reg_pred, reg_target, reg_mask


@BBOX_SAMPLERS.register_module()
class RefinePlanningTargetV4(): ## cls: one pos; reg: all
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        num_cmd=3,
        cls_weight=100.0,
    ):
        super(RefinePlanningTargetV4, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.cls_weight = cls_weight

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        cls_pred = cls_pred[:, 0] # bs, num_mode
        reg_pred = reg_pred[:, 0].cumsum(dim=-2) # bs, num_mode, fut_ts, 2
        gt_reg_target = gt_reg_target.cumsum(dim=-2)
        dist = torch.linalg.norm(gt_reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * gt_reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)

        bs = reg_pred.shape[0]
        num_mode = reg_pred.shape[2]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cls_target = torch.ones([bs, self.ego_fut_mode], dtype=torch.long, device=cls_pred.device)
        cls_target[bs_indices, mode_idx] = 0
        cls_weight = gt_reg_mask.any(dim=-1, keepdim=True) * dist
        cls_weight[bs_indices, mode_idx] = 1
        cls_weight *= self.cls_weight

        reg_target = reg_pred
        reg_mask = reg_pred.new_zeros(reg_pred.shape)[..., 0]

        return cls_pred[..., None], cls_target, cls_weight, reg_pred, reg_target, reg_mask


@BBOX_SAMPLERS.register_module()
class PlanningTargetV5():
    def __init__(
        self,
        point_weight=None,
        point_norm=False,
        hydra_target=False,
        match_by_speed=False,
    ):
        super(PlanningTargetV5, self).__init__()
        self.point_weight = point_weight
        self.point_norm = point_norm
        self.hydra_target = hydra_target
        self.match_by_speed = match_by_speed

    def sample(
        self,
        cls_pred,
        reg_pred,
        reg_target,
        reg_mask,
        cls_target_type,
        mode_idx=None,
        dist=None,
        data=None,
    ):
        bs, num_mode = reg_pred.shape[:2]
        bs_indices = torch.arange(bs, device=reg_pred.device)

        if mode_idx is None:
            mode_idx, dist = self.get_mode_idx(reg_pred, reg_target, reg_mask)

        assert cls_target_type in ["wta", "hydra"]

        if cls_target_type == "wta":
            cls_target = mode_idx
            cls_weight = reg_mask.any(dim=-1)
        elif cls_target_type == "hydra":
            if not self.hydra_target:
                cls_target = torch.ones([bs, num_mode], dtype=torch.long, device=cls_pred.device)
                cls_target[bs_indices, mode_idx] = 0
                cls_weight = reg_mask.any(dim=-1, keepdim=True) * dist
                cls_weight[bs_indices, mode_idx] = 1
                if "data_weight" in data:
                    cls_weight *= data["data_weight"][:,None]

                cls_pred = cls_pred.flatten().unsqueeze(-1)
                cls_target = cls_target.flatten()
                cls_weight = cls_weight.flatten()
            else:
                cls_pred = cls_pred.flatten().unsqueeze(-1)
                cls_target = dist.softmax(dim=-1).flatten()
                cls_weight = reg_mask.any(dim=-1)

        best_reg = reg_pred[bs_indices, mode_idx]

        return cls_pred, cls_target, cls_weight, best_reg, reg_target, reg_mask.unsqueeze(-1)

    def get_mode_idx(self, reg_pred, reg_target, reg_mask, data=None):
        bs, mode, ts, _ = reg_pred.shape
        if self.point_norm:
            r_max = reg_pred.max(dim=1).values.unsqueeze(1)
            r_min = reg_pred.min(dim=1).values.unsqueeze(1)
            delta = torch.abs(reg_target.unsqueeze(1) - reg_pred)
            delta_norm = (delta - r_min) / (r_max - r_min)
            dist = torch.linalg.norm(delta_norm, dim=-1)
        else:
            dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
        if self.point_weight is not None:
            point_weight = dist.new_tensor(self.point_weight)
            dist *= point_weight
        if self.match_by_speed:
            ego_vel = data["ego_status"][:, 6]
            max_dist = ego_vel.round().to(torch.long)
            max_dist = torch.minimum(torch.tensor(14), max_dist)
            max_dist = torch.maximum(torch.tensor(5), max_dist)
            for b in range(bs):
                reg_mask[b, max_dist[b]:] = 0
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist


@BBOX_SAMPLERS.register_module()
class PlanningTargetV6():  ### lon lat decouple
    def __init__(
        self,
        point_weight=None,
        point_norm=False,
        hydra_target=False,
        match_by_speed=False,
        lat_target_mode="norm",
        lon_target_mode="norm",
        lat_chamfer_dist=False,
        lat_mask_cnt=1,
        lon_mask_cnt=1,
        distinct_thresh=0,
        lon_distinct_thresh=0,
        time_points=[],
        collision_sup=False,
        collision_sup_mode=0, # 0: all, 1: wta
    ):
        super(PlanningTargetV6, self).__init__()
        self.point_weight = point_weight
        self.point_norm = point_norm
        self.hydra_target = hydra_target
        self.match_by_speed = match_by_speed
        self.lat_target_mode = lat_target_mode
        self.lon_target_mode = lon_target_mode
        self.lat_chamfer_dist = lat_chamfer_dist
        self.lat_mask_cnt = lat_mask_cnt
        self.lon_mask_cnt = lon_mask_cnt
        self.distinct_thresh = distinct_thresh
        self.lon_distinct_thresh = lon_distinct_thresh
        self.time_points = time_points
        self.collision_sup = collision_sup
        self.collision_sup_mode = collision_sup_mode

    def sample(
        self,
        plan_result,
        data,
        use_lat_query,
        use_lon_query
    ):
        lat_cls = plan_result["spatial_cls"]
        lat_reg = plan_result["spatial_reg"]
        lon_cls = plan_result["vel_seq_cls"]
        lon_reg = plan_result["vel_seq_reg"]

        gt_lat = data["gt_spatial"]
        gt_lat_mask = data["gt_spatial_mask"]
        gt_lon = data["gt_vel_seq"]
        gt_lon_mask = data["gt_vel_seq_mask"]

        lat_mode_idx, lat_dist = self.lat_match(lat_reg, gt_lat, gt_lat_mask)
        lon_mode_idx, lon_dist = self.lon_match(lon_reg, gt_lon, gt_lon_mask)

        bs = lat_cls.shape[0]
        bs_indices = torch.arange(bs, device=lat_cls.device)
        if not use_lat_query:
            lat_cls = lat_cls[bs_indices, lon_mode_idx].unsqueeze(-1)
        if not use_lon_query:
            lon_cls = lon_cls[bs_indices, lat_mode_idx].unsqueeze(-1)

        num_lat_mode = lat_reg.shape[1]
        assert self.lat_target_mode in ["vad", "hydra"]
        if self.lat_target_mode == "norm0": ## bs * num_cls, 1
            lat_cls_target = torch.ones([bs, num_lat_mode], dtype=torch.long, device=lat_cls.device)
            lat_cls_target[bs_indices, lat_mode_idx] = 0
            lat_cls_weight = None

            lat_cls = lat_cls.flatten().unsqueeze(-1)
            lat_cls_target = lat_cls_target.flatten()
        elif self.lat_target_mode == "norm": ## bs, num_cls
            lat_cls_target = lat_mode_idx
            lat_cls = lat_cls.squeeze(-1)
            lat_cls_weight = None
        elif self.lat_target_mode == "vad0":
            lat_cls_target = torch.ones([bs, num_lat_mode], dtype=torch.long, device=lat_cls.device)
            lat_cls_target[bs_indices, lat_mode_idx] = 0
            lat_cls_weight = gt_lat_mask.any(dim=-1, keepdim=True) * lat_dist
            lat_cls_weight[bs_indices, lat_mode_idx] = 1

            lat_cls = lat_cls.flatten().unsqueeze(-1)
            lat_cls_target = lat_cls_target.flatten()
            lat_cls_weight = lat_cls_weight.flatten()
        elif self.lat_target_mode == "vad":
            lat_cls_target = lat_mode_idx
            lat_cls_weight = lat_dist.clone()
            lat_cls_weight = torch.where(lat_cls_weight > self.distinct_thresh, lat_cls_weight, 0)
            lat_cls_weight[bs_indices, lat_mode_idx] = 1
            lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
            lat_cls_weight = lat_cls_weight * lat_mask

            lat_cls = lat_cls.squeeze(-1)
        elif self.lat_target_mode == "hydra":
            lat_cls_target = torch.softmax(-lat_dist, dim=-1)
            lat_cls = lat_cls.squeeze(-1)
            lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
            lat_cls_weight = lat_mask

        num_lon_mode = lon_reg.shape[1]
        # assert self.lon_target_mode == "vad"
        if self.lon_target_mode == "norm": ## bs, num_cls
            lon_cls_target = lon_mode_idx
            lon_cls = lon_cls.squeeze(-1)
            lon_cls_weight = None
        elif self.lon_target_mode == "vad":
            lon_cls_target = lon_mode_idx
            lon_cls_weight = lon_dist.clone()
            lon_cls_weight = torch.where(lon_cls_weight > self.lon_distinct_thresh, lon_cls_weight, 0)
            lon_cls_weight[bs_indices, lon_mode_idx] = 1
            lon_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt).unsqueeze(-1)
            lon_cls_weight = lon_cls_weight * lon_mask

            lon_cls = lon_cls.squeeze(-1)
        elif self.lon_target_mode == "hydra":
            lon_cls_target = torch.softmax(-lon_dist, dim=-1)
            lon_cls = lon_cls.squeeze(-1)
            lon_cls_weight = None

        if self.collision_sup:
            lat_anchor = lat_reg[0]
            lon_speed = lon_reg[0]
            time_points = lon_speed.new_tensor(self.time_points)
            time_interval = time_points[:, 1] - time_points[:, 0]
            lon_dist = (lon_speed * time_interval).cumsum(dim=-1)
            traj_anchor, traj_anchor_mask = interp_anchor_to_traj(lat_anchor, lon_dist)
            col_label = get_col_label(traj_anchor, traj_anchor_mask, data)
            col_label = torch.stack(col_label)
            col_cls = plan_result["col_cls"]
            if self.collision_sup_mode == 0:
                col_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt)[:, None, None]
            elif self.collision_sup_mode == 1:
                col_cls = col_cls[bs_indices, lat_mode_idx]
                col_label = col_label[bs_indices, lat_mode_idx]
                col_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt)[:, None]

        else:
            col_cls = col_label = col_mask = None

        return lat_cls, lat_cls_target, lat_cls_weight, lon_cls, lon_cls_target, lon_cls_weight, col_cls, col_label, col_mask

    def lat_match(self, reg_pred, reg_target, reg_mask):
        bs, mode, ts, _ = reg_pred.shape
        if self.lat_chamfer_dist:
            dist = chamfer_distance(reg_pred, reg_target, reg_mask)
        else:
            dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
            dist = dist * reg_mask.unsqueeze(1)
            dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def lon_match(self, reg_pred, reg_target, reg_mask):
        bs, mode, _ = reg_pred.shape
        dist = torch.abs(reg_target.unsqueeze(1) - reg_pred)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

def chamfer_distance(line1, line2, mask, eps=1e-6):
    '''
    line1: [bs, num_mode, num_pts, 2]
    line2: [bs, num_pts, 2]
    mask: [bs, num_pts]
    '''
    bs, num_mode, num_pts = line1.shape[:3]
    line1 = line1.flatten(0, 1)
    line2 = torch.tile(line2.unsqueeze(1), (1, num_mode, 1, 1)).flatten(0, 1).float()
    mask = torch.tile(mask.unsqueeze(1), (1, num_mode, 1)).flatten(0, 1)

    dist_matrix = torch.cdist(line1, line2, p=2)
    dist_matrix = torch.where(mask.unsqueeze(1).bool(), dist_matrix, 1e6)
    dist12 = dist_matrix.min(dim=2)[0].mean(dim=1)
    dist21 = (dist_matrix.min(dim=1)[0] * mask).sum(-1) / (mask.sum(dim=-1) + eps)
    dist = (dist12 + dist21) / 2
    dist = torch.where(mask.any(dim=-1), dist, 0.)
    dist = dist.unflatten(0, (bs, num_mode))

    return dist



@BBOX_SAMPLERS.register_module()
class PlanningTargetV9():  ### lon lat decouple, sequence + parallel
    def __init__(
        self,
        point_weight=None,
        point_norm=False,
        hydra_target=False,
        match_by_speed=False,
        lat_target_mode="norm",
        lon_target_mode="norm",
        lat_chamfer_dist=False,
        lat_mask_cnt=1,
        lon_mask_cnt=1,
        distinct_thresh=0,
        lon_distinct_thresh=0,
    ):
        super(PlanningTargetV9, self).__init__()
        self.point_weight = point_weight
        self.point_norm = point_norm
        self.hydra_target = hydra_target
        self.match_by_speed = match_by_speed
        self.lat_target_mode = lat_target_mode
        self.lon_target_mode = lon_target_mode
        self.lat_chamfer_dist = lat_chamfer_dist
        self.lat_mask_cnt = lat_mask_cnt
        self.lon_mask_cnt = lon_mask_cnt
        self.distinct_thresh = distinct_thresh
        self.lon_distinct_thresh = lon_distinct_thresh

    def sample(
        self,
        plan_result,
        data,
        use_lat_query,
        use_lon_query
    ):
        lat_cls = plan_result["spatial_cls"]
        lat_reg = plan_result["spatial_reg"]
        lon_cls = plan_result["vel_seq_cls"]
        lon_reg = plan_result["vel_seq_reg"]
        lon_aux_cls = plan_result["vel_seq_aux_cls"]
        lon_aux_reg = plan_result["vel_seq_aux_reg"]

        gt_lat = data["gt_spatial"]
        gt_lat_mask = data["gt_spatial_mask"]
        gt_lon = data["gt_vel_seq"]
        gt_lon_mask = data["gt_vel_seq_mask"]

        lat_mode_idx, lat_dist = self.lat_match(lat_reg, gt_lat, gt_lat_mask)
        lon_mode_idx, lon_dist = self.lon_match(lon_reg, gt_lon, gt_lon_mask)

        bs = lat_cls.shape[0]
        bs_indices = torch.arange(bs, device=lat_cls.device)
        lon_cls = lon_cls[bs_indices, lat_mode_idx]

        num_lat_mode = lat_reg.shape[1]
        lat_cls_target = lat_mode_idx
        lat_cls_weight = lat_dist.clone()
        lat_cls_weight = torch.where(lat_cls_weight > self.distinct_thresh, lat_cls_weight, 0)
        lat_cls_weight[bs_indices, lat_mode_idx] = 1
        lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
        lat_cls_weight = lat_cls_weight * lat_mask
        lat_cls = lat_cls.squeeze(-1)

        lon_cls_target = lon_mode_idx
        lon_cls_weight = lon_dist.clone()
        lon_cls_weight = torch.where(lon_cls_weight > self.lon_distinct_thresh, lon_cls_weight, 0)
        lon_cls_weight[bs_indices, lon_mode_idx] = 1
        lon_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt).unsqueeze(-1)
        lon_cls_weight = lon_cls_weight * lon_mask
        lon_cls = lon_cls.squeeze(-1)
        lon_aux_cls = lon_aux_cls.squeeze(-1)

        return lat_cls, lat_cls_target, lat_cls_weight, lon_cls, lon_aux_cls, lon_cls_target, lon_cls_weight

    def lat_match(self, reg_pred, reg_target, reg_mask):
        bs, mode, ts, _ = reg_pred.shape
        if self.lat_chamfer_dist:
            dist = chamfer_distance(reg_pred, reg_target, reg_mask)
        else:
            dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
            dist = dist * reg_mask.unsqueeze(1)
            dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def lon_match(self, reg_pred, reg_target, reg_mask):
        bs, mode, _ = reg_pred.shape
        dist = torch.abs(reg_target.unsqueeze(1) - reg_pred)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

@BBOX_SAMPLERS.register_module()
class PlanningTargetV6_1():  ### target lon lat combine traj
    def __init__(
        self,
        point_weight=None,
        point_norm=False,
        hydra_target=False,
        match_by_speed=False,
        lat_target_mode="norm",
        lon_target_mode="norm",
        lat_chamfer_dist=False,
        lat_mask_cnt=1,
        lon_mask_cnt=1,
        distinct_thresh=0,
        lon_distinct_thresh=0,
        time_points=[],
        collision_sup=False,
    ):
        super(PlanningTargetV6_1, self).__init__()
        self.point_weight = point_weight
        self.point_norm = point_norm
        self.hydra_target = hydra_target
        self.match_by_speed = match_by_speed
        self.lat_target_mode = lat_target_mode
        self.lon_target_mode = lon_target_mode
        self.lat_chamfer_dist = lat_chamfer_dist
        self.lat_mask_cnt = lat_mask_cnt
        self.lon_mask_cnt = lon_mask_cnt
        self.distinct_thresh = distinct_thresh
        self.lon_distinct_thresh = lon_distinct_thresh
        self.time_points = time_points
        self.collision_sup = collision_sup

    def sample(
        self,
        plan_result,
        data,
        use_lat_query,
        use_lon_query
    ):
        lat_cls = plan_result["spatial_cls"]
        lat_reg = plan_result["spatial_reg"]
        lon_cls = plan_result["vel_seq_cls"]
        lon_reg = plan_result["vel_seq_reg"]

        gt_lat = data["gt_spatial"]
        gt_lat_mask = data["gt_spatial_mask"]
        gt_lon = data["gt_vel_seq"]
        gt_lon_mask = data["gt_vel_seq_mask"]
        gt_traj = data["gt_traj"]
        gt_traj_mask = data["gt_traj_mask"]

        lat_mode_idx, lat_dist = self.lat_match(lat_reg, gt_lat, gt_lat_mask)

        bs = lat_cls.shape[0]
        bs_indices = torch.arange(bs, device=lat_cls.device)
        num_lat_mode = lat_reg.shape[1]
        assert self.lat_target_mode in ["vad", "hydra"]
        if self.lat_target_mode == "norm0": ## bs * num_cls, 1
            lat_cls_target = torch.ones([bs, num_lat_mode], dtype=torch.long, device=lat_cls.device)
            lat_cls_target[bs_indices, lat_mode_idx] = 0
            lat_cls_weight = None

            lat_cls = lat_cls.flatten().unsqueeze(-1)
            lat_cls_target = lat_cls_target.flatten()
        elif self.lat_target_mode == "norm": ## bs, num_cls
            lat_cls_target = lat_mode_idx
            lat_cls = lat_cls.squeeze(-1)
            lat_cls_weight = None
        elif self.lat_target_mode == "vad0":
            lat_cls_target = torch.ones([bs, num_lat_mode], dtype=torch.long, device=lat_cls.device)
            lat_cls_target[bs_indices, lat_mode_idx] = 0
            lat_cls_weight = gt_lat_mask.any(dim=-1, keepdim=True) * lat_dist
            lat_cls_weight[bs_indices, lat_mode_idx] = 1

            lat_cls = lat_cls.flatten().unsqueeze(-1)
            lat_cls_target = lat_cls_target.flatten()
            lat_cls_weight = lat_cls_weight.flatten()
        elif self.lat_target_mode == "vad":
            lat_cls_target = lat_mode_idx
            lat_cls_weight = lat_dist.clone()
            lat_cls_weight = torch.where(lat_cls_weight > self.distinct_thresh, lat_cls_weight, 0)
            lat_cls_weight[bs_indices, lat_mode_idx] = 1
            lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
            lat_cls_weight = lat_cls_weight * lat_mask

            lat_cls = lat_cls.squeeze(-1)
        elif self.lat_target_mode == "hydra":
            lat_cls_target = torch.softmax(-lat_dist, dim=-1)
            lat_cls = lat_cls.squeeze(-1)
            lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
            lat_cls_weight = lat_mask

        lat_anchor = lat_reg[0]
        lon_speed = lon_reg[0]
        time_points = lon_speed.new_tensor(self.time_points)
        time_interval = time_points[:, 1] - time_points[:, 0]
        lon_dist = (lon_speed * time_interval).cumsum(dim=-1)
        traj_anchor, traj_anchor_mask = self.interp_anchor_to_traj(lat_anchor, lon_dist)

        traj_cls = lon_cls.flatten(1, 2)
        lon_reg = lon_reg.unsqueeze(1).repeat(1, num_lat_mode, 1, 1).flatten(1, 2)
        traj_anchor_ = traj_anchor.unsqueeze(0).repeat(bs, 1, 1, 1, 1).flatten(1, 2)
        traj_anchor_mask_ = traj_anchor_mask.unsqueeze(0).repeat(bs, 1, 1, 1).flatten(1, 2)
        mask = traj_anchor_mask_ * gt_traj_mask.unsqueeze(1)
        traj_mode_idx, traj_dist = self.traj_match(traj_anchor_, gt_traj, mask)

        traj_cls_target = traj_mode_idx
        traj_cls_weight = traj_dist.clone()
        # traj_cls_weight = torch.where(traj_cls_weight > self.distinct_thresh, traj_cls_weight, 0)
        traj_cls_weight[bs_indices, traj_mode_idx] = 1
        mask = mask.sum(dim=-1) >= 1
        traj_cls_weight = traj_cls_weight * mask

        if self.collision_sup:
            col_label = get_col_label(traj_anchor, traj_anchor_mask, data)
            col_label = torch.stack(col_label)
            col_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt)[:, None, None]
            col_cls = plan_result["col_cls"]
        else:
            col_cls = col_label = col_mask = None

        return lat_cls, lat_cls_target, lat_cls_weight, traj_cls, traj_cls_target, traj_cls_weight, col_cls, col_label, col_mask

    def lat_match(self, reg_pred, reg_target, reg_mask):
        bs, mode, ts, _ = reg_pred.shape
        if self.lat_chamfer_dist:
            dist = chamfer_distance(reg_pred, reg_target, reg_mask)
        else:
            dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
            dist = dist * reg_mask.unsqueeze(1)
            dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def traj_match(self, reg_pred, reg_target, mask):
        bs, mode = reg_pred.shape[:2]
        dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)

        dist = dist * mask
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def interp_anchor_to_traj(self, spatial_anchor, dist_anchor):
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

    def interp_anchor_to_traj_loop(self, spatial_anchor, dist_anchor):
        """
        spatial_anchor: [N, P, 2]
        dist_anchor:    [M, K]
        return traj:    [N, M, K, 2]
        """
        spatial_anchor = torch.cat([spatial_anchor[:, :1]*0, spatial_anchor], dim=1)
        N, P, _ = spatial_anchor.shape
        M, K = dist_anchor.shape
        # 1. 计算每条 anchor 的累计距离 [N, P]
        seg = torch.diff(spatial_anchor, dim=1)          # [N, P-1, 2]
        seg_len = seg.norm(dim=-1)                       # [N, P-1]
        s_anchor = torch.nn.functional.pad(seg_len.cumsum(dim=1), (1, 0))  # [N, P]

        # 2. 提前申请结果张量
        traj = torch.empty(N, M, K, 2, dtype=spatial_anchor.dtype,
                        device=spatial_anchor.device)

        # 3. 三重 for 循环：样本、距离模式、距离点
        for n in range(N):
            s = s_anchor[n]          # [P]
            xy = spatial_anchor[n]   # [P, 2]
            for m in range(M):
                for k in range(K):
                    q = dist_anchor[m, k].clamp(s[0].item(), s[-1].item())

                    # 找 i 使得 s[i] <= q < s[i+1]
                    i = 0
                    while i < P - 2 and s[i + 1] < q:
                        i += 1

                    # 线性插值
                    ds = s[i + 1] - s[i]
                    w = (q - s[i]) / (ds + 1e-6)
                    traj[n, m, k] = xy[i] + w * (xy[i + 1] - xy[i])

        query = query = dist_anchor.unsqueeze(0).expand(N, -1, -1)
        mask = (query < s_anchor[:, None, -1:]).float()
        return traj, mask

@BBOX_SAMPLERS.register_module()
class UnifiedPlanningTarget():  ### lat lon traj
    def __init__(
        self,
        plan_config=None,
        point_weight=None,
        normalize_points=False,
        sigma=1.0,
    ):
        super(UnifiedPlanningTarget, self).__init__()
        self.plan_config = plan_config
        self.point_weight = point_weight
        self.normalize_points = normalize_points
        self.sigma = sigma

    def sample(
        self,
        plan_result,
        data,
        traj_anchor,
        traj_anchor_mask,
    ):
        ## ------------- lat ------------- ##
        lat_cls = plan_result["lat_cls"].squeeze(-1)
        lat_reg = plan_result["lat_reg"]
        gt_lat = data["gt_lat"]
        gt_lat_mask = data["gt_lat_mask"]
        bs, num_lat_mode = lat_cls.shape
        num_pos = self.plan_config["lat"]["pos"]["num"]
        num_ignore = self.plan_config["lat"]["ignore"]["num"]
        num_neg = num_lat_mode - num_pos - num_ignore

        row_idx = torch.arange(bs, device=lat_cls.device).view(-1, 1).expand(-1, num_lat_mode)
        pos_row_idx = row_idx[:, : num_pos]
        ignore_row_idx = row_idx[:, num_pos : num_pos + num_ignore]
        neg_row_idx = row_idx[:, num_pos + num_ignore :]

        lat_dist = self.lat_match(lat_reg, gt_lat, gt_lat_mask).float()
        sorted_dist, sorted_idx = torch.topk(lat_dist, num_lat_mode, dim=1, largest=False, sorted=True)
        pos_col_idx = sorted_idx[:, : num_pos]
        ignore_col_idx = sorted_idx[:, num_pos : num_pos + num_ignore]
        neg_col_idx = sorted_idx[:, num_pos + num_ignore :]

        ## cache for traj
        lat_row_idx = row_idx.clone()
        lat_sorted_dist = sorted_dist.clone()
        lat_sorted_idx = sorted_idx.clone()

        lat_cls_target = lat_cls.new_zeros(lat_cls.shape)
        lat_cls_weight = lat_cls.new_zeros(lat_cls.shape)

        ## pos
        if self.plan_config["lat"]["pos"]["target_mode"] == "norm":
            pos_target = 1.0
        elif self.plan_config["lat"]["pos"]["target_mode"] == "dist":
            pos_dist = lat_dist[pos_row_idx, pos_col_idx]
            pos_target = torch.softmax(-pos_dist, dim=-1)
        lat_cls_target[pos_row_idx, pos_col_idx] = pos_target

        if self.plan_config["lat"]["pos"]["weight_mode"] == "norm":
            pos_weight = 1.0
        elif self.plan_config["lat"]["pos"]["weight_mode"] == "dist":
            pos_dist = lat_dist[pos_row_idx, pos_col_idx]
            pos_weight = torch.softmax(-pos_dist, dim=-1)
            pos_weight /= pos_weight.mean(dim=-1, keepdims=True)
        lat_cls_weight[pos_row_idx, pos_col_idx] = pos_weight * self.plan_config["lat"]["pos"]["weight"]

        ## neg
        if self.plan_config["lat"]["neg"]["weight_mode"] == "norm":
            neg_dist = lat_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist.mean(dim=-1,keepdim=True)
        elif self.plan_config["lat"]["neg"]["weight_mode"] == "dist":
            neg_dist = lat_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist
        lat_cls_weight[neg_row_idx, neg_col_idx] = neg_weight * self.plan_config["lat"]["neg"]["weight"]

        lat_mask = (gt_lat_mask.sum(dim=1) >= 1).unsqueeze(-1)
        lat_cls_weight = lat_cls_weight * lat_mask


        ## ------------- lon ------------- ##
        lon_cls = plan_result["lon_cls"].squeeze(-1)
        lon_reg = plan_result["lon_reg"]
        gt_lon = data["gt_lon"]
        gt_lon_mask = data["gt_lon_mask"]
        num_lon_mode = lon_cls.shape[1]
        num_pos = self.plan_config["lon"]["pos"]["num"]
        num_ignore = self.plan_config["lon"]["ignore"]["num"]
        num_neg = num_lon_mode - num_pos - num_ignore

        row_idx = torch.arange(bs, device=lon_cls.device).view(-1, 1).expand(-1, num_lon_mode)
        pos_row_idx = row_idx[:, : num_pos]
        ignore_row_idx = row_idx[:, num_pos : num_pos + num_ignore]
        neg_row_idx = row_idx[:, num_pos + num_ignore :]

        lon_dist = self.lon_match(lon_reg, gt_lon, gt_lon_mask).float()
        sorted_dist, sorted_idx = torch.topk(lon_dist, num_lon_mode, dim=1, largest=False, sorted=True)
        pos_col_idx = sorted_idx[:, : num_pos]
        ignore_col_idx = sorted_idx[:, num_pos : num_pos + num_ignore]
        neg_col_idx = sorted_idx[:, num_pos + num_ignore :]

        ## cache for traj
        lon_row_idx = row_idx.clone()
        lon_sorted_dist = sorted_dist.clone()
        lon_sorted_idx = sorted_idx.clone()

        lon_cls_target = lon_cls.new_zeros(lon_cls.shape)
        lon_cls_weight = lon_cls.new_zeros(lon_cls.shape)

        ## pos
        if self.plan_config["lon"]["pos"]["target_mode"] == "norm":
            pos_target = 1.0
        elif self.plan_config["lon"]["pos"]["target_mode"] == "dist":
            pos_dist = lon_dist[pos_row_idx, pos_col_idx]
            pos_target = torch.softmax(-pos_dist, dim=-1)
        lon_cls_target[pos_row_idx, pos_col_idx] = pos_target

        if self.plan_config["lon"]["pos"]["weight_mode"] == "norm":
            pos_weight = 1.0
        elif self.plan_config["lon"]["pos"]["weight_mode"] == "dist":
            pos_dist = lon_dist[pos_row_idx, pos_col_idx]
            pos_weight = torch.softmax(-pos_dist, dim=-1)
        lon_cls_weight[pos_row_idx, pos_col_idx] = pos_weight * self.plan_config["lon"]["pos"]["weight"]

        ## neg
        if self.plan_config["lon"]["neg"]["weight_mode"] == "norm":
            neg_weight = 1.0
        elif self.plan_config["lon"]["neg"]["weight_mode"] == "dist":
            neg_dist = lon_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist
        lon_cls_weight[neg_row_idx, neg_col_idx] = neg_weight * self.plan_config["lon"]["neg"]["weight"]

        lon_mask = (gt_lon_mask.sum(dim=1) >= 1).unsqueeze(-1)
        lon_cls_weight = lon_cls_weight * lon_mask

        ## ------------- traj ------------- ##
        traj_cls = plan_result["traj_cls"].squeeze(-1)
        traj_reg = plan_result["traj_reg"]
        traj_reg_mask = plan_result["traj_reg_mask"]
        gt_traj = data["gt_traj"]
        gt_traj_mask = data["gt_traj_mask"]
        traj_dist = self.traj_match(traj_reg, traj_reg_mask, gt_traj, gt_traj_mask).float()

        traj_cls = traj_cls.flatten(1, 2)
        traj_reg = traj_reg.flatten(1, 2)
        traj_dist = traj_dist.flatten(1, 2)

        bs, num_traj_mode = traj_cls.shape
        num_pos = self.plan_config["traj"]["pos"]["num"]
        num_ignore = self.plan_config["traj"]["ignore"]["num"]
        num_neg = num_traj_mode - num_pos - num_ignore

        row_idx = torch.arange(bs, device=traj_cls.device).view(-1, 1).expand(-1, num_traj_mode)
        pos_row_idx = row_idx[:, : num_pos]
        ignore_row_idx = row_idx[:, num_pos : num_pos + num_ignore]
        neg_row_idx = row_idx[:, num_pos + num_ignore :]

        sorted_dist, sorted_idx = torch.topk(traj_dist, num_traj_mode, dim=1, largest=False, sorted=True)
        pos_col_idx = sorted_idx[:, : num_pos]
        ignore_col_idx = sorted_idx[:, num_pos : num_pos + num_ignore]
        neg_col_idx = sorted_idx[:, num_pos + num_ignore :]

        traj_cls_target = traj_cls.new_zeros(traj_cls.shape)
        traj_cls_weight = traj_cls.new_zeros(traj_cls.shape)

        ## pos
        if self.plan_config["traj"]["pos"]["target_mode"] == "norm":
            pos_target = 1.0
        elif self.plan_config["traj"]["pos"]["target_mode"] == "dist":
            pos_dist = traj_dist[pos_row_idx, pos_col_idx]
            pos_target = torch.softmax(-pos_dist  / self.sigma, dim=-1)
        elif self.plan_config["traj"]["pos"]["target_mode"] == "func0":
            pos_dist = traj_dist[pos_row_idx, pos_col_idx]
            pos_target = torch.exp(- (pos_dist ** 2) )

        traj_cls_target[pos_row_idx, pos_col_idx] = pos_target

        if self.plan_config["traj"]["pos"]["weight_mode"] == "norm":
            pos_weight = 1.0
        elif self.plan_config["traj"]["pos"]["weight_mode"] == "dist":
            pos_dist = traj_dist[pos_row_idx, pos_col_idx]
            pos_weight = torch.softmax(-pos_dist, dim=-1)
            pos_weight /= pos_weight.mean(dim=-1, keepdims=True)
        traj_cls_weight[pos_row_idx, pos_col_idx] = pos_weight * self.plan_config["traj"]["pos"]["weight"]

        ## neg
        if self.plan_config["traj"]["neg"]["weight_mode"] == "norm":
            neg_dist = traj_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist.mean(dim=-1,keepdim=True)
        elif self.plan_config["traj"]["neg"]["weight_mode"] == "dist":
            neg_dist = traj_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist
        traj_cls_weight[neg_row_idx, neg_col_idx] = neg_weight * self.plan_config["traj"]["neg"]["weight"]

        mask = traj_reg_mask * gt_traj_mask[:, None, None]
        traj_mask = (mask.sum(dim=-1) >= 1).flatten(1, 2)
        traj_cls_weight = traj_cls_weight * traj_mask

        if "lat_pos_num" in self.plan_config["traj"]:
            lat_pos_num = self.plan_config["traj"]["lat_pos_num"]
            pos_row_idx = lat_row_idx[:, : lat_pos_num]
            pos_col_idx = lat_sorted_idx[:, : lat_pos_num]
            lat_pos_mask = traj_cls.new_zeros((bs, num_lat_mode, num_lon_mode))
            lat_pos_mask[pos_row_idx, pos_col_idx] = 1.0
            traj_cls_weight *= lat_pos_mask.flatten(1, 2)

            # traj_cls = traj_cls.unflatten(1, (num_lat_mode, num_lon_mode))[pos_row_idx, pos_col_idx].flatten(1, 2)
            # traj_cls_target = traj_cls_target.unflatten(1, (num_lat_mode, num_lon_mode))[pos_row_idx, pos_col_idx].flatten(1, 2)
            # traj_cls_weight = traj_cls_weight.unflatten(1, (num_lat_mode, num_lon_mode))[pos_row_idx, pos_col_idx].flatten(1, 2)

        if "lon_pos_num" in self.plan_config["traj"]:
            lon_pos_num = self.plan_config["traj"]["lon_pos_num"]
            pos_row_idx = lon_row_idx[:, : lon_pos_num]
            pos_col_idx = lon_sorted_idx[:, : lon_pos_num]
            lon_pos_mask = traj_cls.new_zeros((bs, num_lat_mode, num_lon_mode))
            lon_pos_mask[pos_row_idx, :, pos_col_idx] = 1.0
            traj_cls_weight *= lon_pos_mask.flatten(1, 2)

        ## col
        if "collision_cls" in plan_result:
            col_cls = plan_result["collision_cls"].squeeze(-1)
            col_label = get_col_label(traj_anchor, traj_anchor_mask, data)
            col_label = torch.stack(col_label).to(col_cls.dtype)
        else:
            col_cls = col_label = None

        return (
            lat_cls, lat_cls_target, lat_cls_weight,
            lon_cls, lon_cls_target, lon_cls_weight,
            traj_cls, traj_cls_target, traj_cls_weight,
            col_cls, col_label
        )

    def lat_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        return dist

    def lon_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.abs(reg_target.unsqueeze(1) - reg_pred)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        return dist

    def traj_match(self, traj_reg, traj_reg_mask, gt_traj, gt_traj_mask, eps=1e-7):
        if self.normalize_points:
            r_max = traj_reg.flatten(1, 2).max(dim=1).values.unsqueeze(1).unsqueeze(1)
            r_min = traj_reg.flatten(1, 2).min(dim=1).values.unsqueeze(1).unsqueeze(1)
            norm_reg = (traj_reg - r_min) / (r_max - r_min + eps)
            norm_gt = (gt_traj[:, None, None] - r_min) / (r_max - r_min + eps)
            dist = torch.linalg.norm(norm_reg - norm_gt, dim=-1)
        else:
            dist = torch.linalg.norm(gt_traj[:, None, None] - traj_reg, dim=-1)
        if self.point_weight is not None:
            point_weight = dist.new_tensor(self.point_weight)
            dist *= point_weight
        gt_traj_mask = gt_traj_mask[:, None, None]
        mask = traj_reg_mask * gt_traj_mask
        dist = (dist * mask).sum(dim=-1)
        dist = dist / (mask.sum(dim=-1) + eps)
        return dist


@BBOX_SAMPLERS.register_module()
class TemporalTrajPlanningTarget():  ### temporal traj
    def __init__(
        self,
        plan_config=None,
    ):
        super(TemporalTrajPlanningTarget, self).__init__()
        self.plan_config = plan_config

    def sample(
        self,
        plan_result,
        data,
    ):
        ## ------------- traj ------------- ##
        traj_cls = plan_result["traj_cls"].squeeze(-1)
        traj_reg = plan_result["traj_reg"]
        gt_traj = data["gt_traj"]
        gt_traj_mask = data["gt_traj_mask"]
        traj_dist = self.traj_match(traj_reg, gt_traj, gt_traj_mask).float()

        bs, num_traj_mode = traj_cls.shape
        num_pos = self.plan_config["traj"]["pos"]["num"]
        num_ignore = self.plan_config["traj"]["ignore"]["num"]
        num_neg = num_traj_mode - num_pos - num_ignore

        row_idx = torch.arange(bs, device=traj_cls.device).view(-1, 1).expand(-1, num_traj_mode)
        pos_row_idx = row_idx[:, : num_pos]
        ignore_row_idx = row_idx[:, num_pos : num_pos + num_ignore]
        neg_row_idx = row_idx[:, num_pos + num_ignore :]

        sorted_dist, sorted_idx = torch.topk(traj_dist, num_traj_mode, dim=1, largest=False, sorted=True)
        pos_col_idx = sorted_idx[:, : num_pos]
        ignore_col_idx = sorted_idx[:, num_pos : num_pos + num_ignore]
        neg_col_idx = sorted_idx[:, num_pos + num_ignore :]

        traj_cls_target = traj_cls.new_zeros(traj_cls.shape)
        traj_cls_weight = traj_cls.new_zeros(traj_cls.shape)

        ## pos
        if self.plan_config["traj"]["pos"]["target_mode"] == "norm":
            pos_target = 1.0
        elif self.plan_config["traj"]["pos"]["target_mode"] == "dist":
            pos_dist = traj_dist[pos_row_idx, pos_col_idx]
            pos_target = torch.softmax(-pos_dist, dim=-1)
        traj_cls_target[pos_row_idx, pos_col_idx] = pos_target

        if self.plan_config["traj"]["pos"]["weight_mode"] == "norm":
            pos_weight = 1.0
        elif self.plan_config["traj"]["pos"]["weight_mode"] == "dist":
            pos_dist = traj_dist[pos_row_idx, pos_col_idx]
            pos_weight = torch.softmax(-pos_dist, dim=-1)
            pos_weight /= pos_weight.mean(dim=-1, keepdims=True)
        traj_cls_weight[pos_row_idx, pos_col_idx] = pos_weight * self.plan_config["traj"]["pos"]["weight"]

        ## neg
        if self.plan_config["traj"]["neg"]["weight_mode"] == "norm":
            neg_dist = traj_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist.mean(dim=-1,keepdim=True)
        elif self.plan_config["traj"]["neg"]["weight_mode"] == "dist":
            neg_dist = traj_dist[neg_row_idx, neg_col_idx]
            neg_weight = neg_dist
        traj_cls_weight[neg_row_idx, neg_col_idx] = neg_weight * self.plan_config["traj"]["neg"]["weight"]

        traj_mask = (gt_traj_mask.sum(dim=1) >= 1).unsqueeze(-1)
        traj_cls_weight = traj_cls_weight * traj_mask

        return traj_cls, traj_cls_target, traj_cls_weight

    def traj_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        return dist

#############################################3

@BBOX_SAMPLERS.register_module()
class V13PlanningTarget():
    def __init__(
        self,
        plan_config=None,
        point_weight=None,
        normalize_points=False,
        lat_mask_cnt=1,
        lon_mask_cnt=1,
        traj_mask_cnt=1,
    ):
        super(V13PlanningTarget, self).__init__()
        self.plan_config = plan_config
        self.point_weight = point_weight
        self.normalize_points = normalize_points

        self.lat_mask_cnt = lat_mask_cnt
        self.lon_mask_cnt = lon_mask_cnt
        self.traj_mask_cnt = traj_mask_cnt

    def sample(
        self,
        plan_result,
        data,
    ):
        ## ------------- lat ------------- ##
        bs = plan_result["lat_cls"].shape[0]
        bs_indices = torch.arange(bs, device=plan_result["lat_cls"].device)

        ## lat
        lat_cls = plan_result["lat_cls"]
        lat_anchor = plan_result["lat_anchor"]
        gt_lat = data["gt_lat"]
        gt_lat_mask = data["gt_lat_mask"]
        lat_mode_idx, lat_dist = self.lat_match(lat_anchor, gt_lat, gt_lat_mask)

        lat_cls_target = lat_mode_idx
        lat_cls_weight = lat_dist.clone()
        lat_cls_weight[bs_indices, lat_mode_idx] = 1
        lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
        lat_cls_weight = lat_cls_weight * lat_mask

        ## lon
        lon_cls = plan_result["lon_cls"]
        lon_anchor = plan_result["lon_anchor"]
        gt_lon = data["gt_lon"]
        gt_lon_mask = data["gt_lon_mask"]
        lon_mode_idx, lon_dist = self.lon_match(lon_anchor, gt_lon, gt_lon_mask)

        lon_cls_target = lon_mode_idx
        lon_cls_weight = lon_dist.clone()
        lon_cls_weight[bs_indices, lon_mode_idx] = 1
        lon_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt).unsqueeze(-1)
        lon_cls_weight = lon_cls_weight * lon_mask

        ## traj
        if "traj_cls" in plan_result:
            traj_cls = plan_result["traj_cls"]
            traj_reg = plan_result["traj_reg"]
            traj_reg_mask = plan_result["traj_reg_mask"]
            gt_traj = data["gt_traj"]
            gt_traj_mask = data["gt_traj_mask"]
            traj_mode_idx, traj_dist = self.traj_match(traj_reg, traj_reg_mask, gt_traj, gt_traj_mask)

            traj_cls_target = traj_mode_idx
            traj_cls_weight = traj_dist.clone()
            traj_cls_weight[bs_indices, traj_mode_idx] = 1
            traj_mask = (gt_traj_mask.sum(dim=1) >= self.traj_mask_cnt).unsqueeze(-1)
            traj_cls_weight = traj_cls_weight * traj_mask
        else:
            traj_cls = traj_cls_target = traj_cls_weight = None

        ## col
        if "collision_cls" in plan_result or "point_collision_cls" in plan_result:
            point_col_label = get_col_label_bs(traj_reg, traj_reg_mask, data).to(traj_cls.dtype)

        if "collision_cls" in plan_result:
            col_cls = plan_result["collision_cls"].squeeze(-1)
            col_label = point_col_label.any(dim=-1).to(traj_cls.dtype)
        else:
            col_cls = col_label = None

        if "point_collision_cls" in plan_result:
            point_col_cls = plan_result["point_collision_cls"]
            point_col_label = point_col_label.to(traj_cls.dtype)
        else:
            point_col_cls = point_col_label = None

        return (
            lat_cls, lat_cls_target, lat_cls_weight,
            lon_cls, lon_cls_target, lon_cls_weight,
            traj_cls, traj_cls_target, traj_cls_weight,
            col_cls, col_label, point_col_cls, point_col_label
        )

    def lat_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def lon_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.abs(reg_target.unsqueeze(1) - reg_pred)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def traj_match(self, traj_reg, traj_reg_mask, gt_traj, gt_traj_mask, eps=1e-7):
        if self.normalize_points:
            r_max = traj_reg.flatten(1, 2).max(dim=1).values.unsqueeze(1).unsqueeze(1)
            r_min = traj_reg.flatten(1, 2).min(dim=1).values.unsqueeze(1).unsqueeze(1)
            norm_reg = (traj_reg - r_min) / (r_max - r_min + eps)
            norm_gt = (gt_traj[:, None, None] - r_min) / (r_max - r_min + eps)
            dist = torch.linalg.norm(norm_reg - norm_gt, dim=-1)
        else:
            dist = torch.linalg.norm(gt_traj.unsqueeze(1) - traj_reg, dim=-1)
        if self.point_weight is not None:
            point_weight = dist.new_tensor(self.point_weight)
            dist *= point_weight
        mask = gt_traj_mask.unsqueeze(1)
        dist = (dist * mask).mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist


@BBOX_SAMPLERS.register_module()
class V13PlanningTargetHydra():
    def __init__(
        self,
        plan_config=None,
        point_weight=None,
        normalize_points=False,
        lat_mask_cnt=1,
        lon_mask_cnt=1,
        traj_mask_cnt=1,
        sigma=1.0
    ):
        super(V13PlanningTargetHydra, self).__init__()
        self.plan_config = plan_config
        self.point_weight = point_weight
        self.normalize_points = normalize_points

        self.lat_mask_cnt = lat_mask_cnt
        self.lon_mask_cnt = lon_mask_cnt
        self.traj_mask_cnt = traj_mask_cnt
        self.sigma = sigma

    def sample(
        self,
        plan_result,
        data,
    ):
        ## ------------- lat ------------- ##
        bs = plan_result["lat_cls"].shape[0]
        bs_indices = torch.arange(bs, device=plan_result["lat_cls"].device)

        ## lat
        lat_cls = plan_result["lat_cls"]
        lat_anchor = plan_result["lat_anchor"]
        gt_lat = data["gt_lat"]
        gt_lat_mask = data["gt_lat_mask"]
        lat_mode_idx, lat_dist = self.lat_match(lat_anchor, gt_lat, gt_lat_mask)

        lat_cls_target = lat_mode_idx
        lat_cls_weight = lat_dist.clone()
        lat_cls_weight[bs_indices, lat_mode_idx] = 1
        lat_mask = (gt_lat_mask.sum(dim=1) >= self.lat_mask_cnt).unsqueeze(-1)
        lat_cls_weight = lat_cls_weight * lat_mask

        ## lon
        lon_cls = plan_result["lon_cls"]
        lon_anchor = plan_result["lon_anchor"]
        gt_lon = data["gt_lon"]
        gt_lon_mask = data["gt_lon_mask"]
        lon_mode_idx, lon_dist = self.lon_match(lon_anchor, gt_lon, gt_lon_mask)

        lon_cls_target = lon_mode_idx
        lon_cls_weight = lon_dist.clone()
        lon_cls_weight[bs_indices, lon_mode_idx] = 1
        lon_mask = (gt_lon_mask.sum(dim=1) >= self.lon_mask_cnt).unsqueeze(-1)
        lon_cls_weight = lon_cls_weight * lon_mask

        ## traj
        if "traj_cls" in plan_result:
            traj_cls = plan_result["traj_cls"]
            traj_reg = plan_result["traj_reg"]
            traj_reg_mask = plan_result["traj_reg_mask"]
            gt_traj = data["gt_traj"]
            gt_traj_mask = data["gt_traj_mask"]
            traj_mode_idx, traj_dist = self.traj_match(traj_reg, traj_reg_mask, gt_traj, gt_traj_mask)
            traj_cls_target = (-traj_dist).softmax(1)
            traj_cls_weight = (gt_traj_mask.sum(dim=1) >= self.traj_mask_cnt).unsqueeze(-1)
        else:
            traj_cls = traj_cls_target = traj_cls_weight = None

        ## col
        if "collision_cls" in plan_result or "point_collision_cls" in plan_result:
            point_col_label = get_col_label_bs(traj_reg, traj_reg_mask, data).to(traj_cls.dtype)

        if "collision_cls" in plan_result:
            col_cls = plan_result["collision_cls"].squeeze(-1)
            col_label = point_col_label.any(dim=-1).to(traj_cls.dtype)
        else:
            col_cls = col_label = None

        if "point_collision_cls" in plan_result:
            point_col_cls = plan_result["point_collision_cls"]
            point_col_label = point_col_label.to(traj_cls.dtype)
        else:
            point_col_cls = point_col_label = None

        return (
            lat_cls, lat_cls_target, lat_cls_weight,
            lon_cls, lon_cls_target, lon_cls_weight,
            traj_cls, traj_cls_target, traj_cls_weight,
            col_cls, col_label, point_col_cls, point_col_label
        )

    def lat_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_pred, dim=-1)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def lon_match(self, reg_pred, reg_target, reg_mask):
        dist = torch.abs(reg_target.unsqueeze(1) - reg_pred)
        dist = dist * reg_mask.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist

    def traj_match(self, traj_reg, traj_reg_mask, gt_traj, gt_traj_mask, eps=1e-7):
        # if self.normalize_points:
        #     r_max = traj_reg.flatten(1, 2).max(dim=1).values.unsqueeze(1).unsqueeze(1)
        #     r_min = traj_reg.flatten(1, 2).min(dim=1).values.unsqueeze(1).unsqueeze(1)
        #     norm_reg = (traj_reg - r_min) / (r_max - r_min + eps)
        #     norm_gt = (gt_traj[:, None, None] - r_min) / (r_max - r_min + eps)
        #     dist = torch.linalg.norm(norm_reg - norm_gt, dim=-1)
        # else:
        num_pts = gt_traj.shape[1]
        dist = gt_traj.unsqueeze(1) - traj_reg
        dist = dist.pow(2).sum(-1)  # [B, M, T]
        mask = gt_traj_mask[:, None].float()
        dist = dist * mask
        valid_cnt = mask.sum(-1).clamp(min=1.0)   # [B, 1]
        dist = dist.sum(-1) / valid_cnt            # [B, 1024]
        dist = dist * self.sigma * num_pts
        mode_idx = torch.argmin(dist, dim=-1)
        return mode_idx, dist
