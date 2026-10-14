from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.datasets.b2d_3d_dataset import Discrete_Actions_DICT
from .rescore_utils import *

@BBOX_CODERS.register_module()
class SparseBox3DMotionDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__()

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        anchor_queue = motion_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = motion_output["period"]

        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            trajs = motion_output["prediction"][-1]
            traj_cls = motion_output["classification"][-1].sigmoid()
            traj = trajs[i, indices[i] // num_cls]
            traj_cls = traj_cls[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]
            output.append(
                {
                    "trajs_3d": traj.cpu(),
                    "trajs_score": traj_cls.cpu()
                }
            )

            temp_anchor = anchor_queue[i, indices[i] // num_cls]
            temp_period = period[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]
            num_pred, queue_len = temp_anchor.shape[:2]
            temp_anchor = temp_anchor.flatten(0, 1)
            temp_anchor = decode_box(temp_anchor)
            temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])
            output[-1]['anchor_queue'] = temp_anchor.cpu()
            output[-1]['period'] = temp_period.cpu()

        return output


@BBOX_CODERS.register_module()
class HierarchicalPlanningDecoder(object):
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
        num_cmd=3,
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore
        self.num_cmd = num_cmd

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        paths = planning_output["paths"][-1]
        bs = classification.shape[0]
        classification = classification.reshape(bs, self.num_cmd, self.ego_fut_mode)
        prediction = prediction.reshape(bs, self.num_cmd, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)
        classification, final_planning = self.select(det_output, motion_output, classification, prediction, data)
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append(
                {
                    "planning_score": cls.sigmoid().cpu(),
                    "planning": pred.cpu(),
                    "final_planning": final_planning[i].cpu(),
                    "ego_period": period[i].cpu(),
                    "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
                }
            )
            if paths is not None:
                output[-1]["path"] = paths[i].reshape(-1,2).cumsum(dim=0).cpu()

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]

        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        if self.num_cmd == 1:
            cmd *= 0

        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        self.use_rescore = False
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        ## max
        mode_idx = plan_cls.argmax(dim=-1)
        ## random
        # prob = torch.softmax(plan_cls.sigmoid(), dim=1)
        # prob = plan_cls.sigmoid()
        # mode_idx = torch.multinomial(prob, num_samples=1, replacement=True).squeeze(-1)  # (bs,)

        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self,
        plan_cls,
        plan_reg,
        motion_cls,
        motion_reg,
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat

        def get_yaw(traj, start_yaw=np.pi/2):
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
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        # ego_box[0] += offset * torch.cos(ego_box[6])
        # ego_box[1] += offset * torch.sin(ego_box[6])
        ego_box[..., 0] += offset * torch.cos(ego_box[..., 6])
        ego_box[..., 1] += offset * torch.sin(ego_box[..., 6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


@BBOX_CODERS.register_module()
class MultiPredPlanningDecoder(object):
    def __init__(
        self,
        plan_config={},
        anchor_reference_group=None,
        use_rescore=False,
    ):
        super(MultiPredPlanningDecoder, self).__init__()
        self.plan_config = plan_config
        self.anchor_reference_group = anchor_reference_group
        self.use_rescore = use_rescore

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        if len(planning_output["planning_refine_results"]) > 0:
            planning_results = planning_output["planning_refine_results"][-1]
        else:
            planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        for key, value in self.plan_config.items():
            if "spatial" in key or "temporal" in key:
                cls = planning_results[key + "_cls"].sigmoid()
                reg = planning_results[key + "_reg"]
                mode_idx = cls.argmax(dim=-1)
                for b in range(bs):
                    output[b][key + "_cls"] = cls[b].cpu()
                    output[b][key + "_reg"] = reg[b].cpu()
                    output[b][key + "_reg_final"] = reg[b, mode_idx[b]].cpu()
            elif "speed" in key:
                ref_cls = planning_results[self.anchor_reference_group + "_cls"].sigmoid()
                mode_idx = ref_cls.argmax(dim=-1)
                cls = planning_results[key + "_cls"].softmax(dim=-1)
                bs = cls.shape[0]
                bs_indices = torch.arange(bs, device=cls.device)
                cls_score = cls[bs_indices, mode_idx]

                ## argmax
                best_cls = cls_score.argmax(dim=-1)
                for b in range(bs):
                    output[b][key] = torch.tensor(value["speed_intervals"][best_cls[b]]).cpu()
                    output[b][key+"_score"] = torch.tensor(cls_score[b]).cpu()
                ## mean
                # speed_reg = (best_cls * torch.tensor(value["speed_intervals"]).to(cls.device)).sum(dim=-1)
                # for b in range(bs):
                #     output[b][key] = speed_reg[b].cpu()
            elif "control" in key:
                control_cls = planning_results["control_cls"][:, 0]
                action_id = control_cls.argmax(dim=-1).cpu().numpy()
                for b in range(bs):
                    output[b]["control"] = Discrete_Actions_DICT[action_id[b]]
            elif "lateral_point" in key:
                ref_cls = planning_results[self.anchor_reference_group + "_cls"].sigmoid()
                ref_reg = planning_results[self.anchor_reference_group + "_reg"]
                mode_idx = ref_cls.argmax(dim=-1)
                cls = planning_results[key + "_cls"]
                bs = cls.shape[0]
                bs_indices = torch.arange(bs, device=cls.device)
                best_cls = cls[bs_indices, mode_idx].softmax(dim=-1)
                best_cls = best_cls.argmax(dim=-1)
                best_reg = ref_reg[bs_indices, mode_idx]
                for b in range(bs):
                    output[b][key] = torch.tensor(best_reg[b, best_cls[b]]).cpu()
            elif "vel_seq" in key:
                if not hasattr(self, f"{key}_anchor"):
                    setattr(self, f"{key}_anchor", np.load(value["anchor"]))

                anchor = getattr(self, f"{key}_anchor")

                ref_cls = planning_results[self.anchor_reference_group + "_cls"].sigmoid()
                mode_idx = ref_cls.argmax(dim=-1)
                cls = planning_results[key + "_cls"].softmax(dim=-1)
                bs = cls.shape[0]
                bs_indices = torch.arange(bs, device=cls.device)
                cls_score = cls[bs_indices, mode_idx]
                ref_reg = planning_results[self.anchor_reference_group + "_reg"]
                ref_reg = ref_reg[bs_indices, mode_idx]

                # def gather_top100(score: torch.Tensor, anchor: torch.Tensor):
                #     """
                #     score  : (bs, num_mode)   分类得分
                #     anchor : (num_mode, N)    与模态对应的 anchor 向量
                #     return :
                #         top100_score   : (100,)        得分
                #         top100_anchor  : (100, N)      对应 anchor
                #     """
                #     anchor=torch.tensor(anchor).to(score.device)
                #     bs, num_mode = score.shape
                #     device = score.device

                #     # 1. 扁平化后取 top-100
                #     flat_score = score.reshape(-1)                      # (bs*num_mode,)
                #     top100_val, top100_idx = torch.topk(flat_score, k=100, largest=True, sorted=True)

                #     # 2. 解耦成 (batch_idx, mode_idx)
                #     batch_idx = top100_idx // num_mode                  # (100,)
                #     mode_idx  = top100_idx %  num_mode                  # (100,)

                #     # 3. 一次性取 anchor
                #     top100_anchor = anchor[mode_idx]                    # (100, N)

                #     return top100_val, top100_anchor

                # cls_score, anchor = gather_top100(cls_score, anchor)
                # cls_score = cls_score[None]

                # cls_score = self.rescore_path(cls_score, anchor, ref_reg, det_output, motion_output, data)

                ## argmax
                best_cls = cls_score.argmax(dim=-1)
                for b in range(bs):
                    output[b][key] = torch.tensor(anchor[best_cls[b].cpu()])

        return output

    def rescore_path(
        self,
        speed_score,
        speed_anchor,
        path_anchor,
        det_output,
        motion_output,
        data,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat

        def get_yaw(traj, start_yaw=np.pi/2):
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

        ### vis traj
        # n=3
        # speed_anchor = speed_anchor[:n]
        # traj = speed_to_trajectory(path_anchor.cpu(), torch.tensor(speed_anchor), dt=0.5)
        # import matplotlib.pyplot as plt
        # for i in range(n):
        #     plt.plot(traj[0,i,:,0]+(i+1), traj[0,i,:,1], marker='o', markersize=7)
        # plt.plot(path_anchor.cpu().numpy()[0,:,0], path_anchor.cpu().numpy()[0,:,1], marker='o', markersize=7)
        # plt.axis('equal')
        # plt.savefig("traj")
        # plan_reg_ = speed_to_trajectory(path_anchor, torch.tensor(speed_anchor).to(path_anchor.device), dt=0.5)[:, :, :6]
        import time
        s1 = time.time()
        plan_reg = speed_to_trajectory_vec(path_anchor, torch.tensor(speed_anchor).to(path_anchor.device), dt=0.5)[:, :, :6].float()
        s2 = time.time()
        print("interp:", s2-s1)

        plan_cls = speed_score
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]
        self.ego_fut_mode = 100
        self.ego_fut_ts = 6
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        s3 = time.time()
        print("prep:", s3-s2)
        # ego_box[0] += offset * torch.cos(ego_box[6])
        # ego_box[1] += offset * torch.sin(ego_box[6])
        ego_box[..., 0] += offset * torch.cos(ego_box[..., 6])
        ego_box[..., 1] += offset * torch.sin(ego_box[..., 6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        s4 = time.time()
        print("rescore:", s4-s3)
        return plan_cls


def path_to_arc_length(path: torch.Tensor) -> torch.Tensor:
    """
    path: (B, P, 2)  二维坐标
    return: (B, P)   从起点到每个点的累计弧长（0 开始）
    """
    path = torch.cat([
        torch.zeros(path.size(0), 1, 2, device=path.device, dtype=path.dtype),
        path,
    ],dim=1)
    seg = torch.diff(path, dim=1)               # (B, P-1, 2)
    seg_len = torch.linalg.norm(seg, dim=-1)    # (B, P-1)
    arc = torch.cumsum(seg_len, dim=1)
    return arc                                    # (B, P)

def interp1d_linear(t: torch.Tensor,         # 查询点 (Q,)
                    x: torch.Tensor,         # 原始横轴 (P,)
                    y: torch.Tensor) -> torch.Tensor:
    """
    1-D 线性插值：y(t)
    t 必须在 [x[0], x[-1]] 内
    return: (Q,)
    """
    idx = torch.searchsorted(x, t, right=False)        # (Q,)
    idx = idx.clamp(1, x.numel() - 1)                  # 保证有左邻居
    x0, x1 = x[idx-1], x[idx]                          # (Q,)
    y0, y1 = y[idx-1], y[idx]
    w = (t - x0) / (x1 - x0 + 1e-8)
    return y0 + w * (y1 - y0)

def speed_to_trajectory(
        path: torch.Tensor,               # (B, P, 2)
        speed: torch.Tensor,              # (M, T)
        dt: float = 0.5) -> torch.Tensor:
    """
    主函数
    path : B 条等距路径，每条 P 个点
    speed: M 种模态，每模态 T 段常数速度（m/s），每段时长 dt
    return: (B, M, T, 2)  每个 batch、每种模态、每个时刻末端的 (x,y)
    """
    B, P, _ = path.shape
    M, T = speed.shape
    device = path.device
    dtype = path.dtype

    # 1. 累计弧长
    arc = path_to_arc_length(path)          # (B, P)

    # 2. 累计位移（每段位移 = v * dt）
    disp = speed * dt                       # (M, T)
    cum_disp = torch.cumsum(disp, dim=1)    # (M, T)

    norm_arc = arc                # (B, P)
    norm_disp = cum_disp # (M, T)

    # 4. 对每条路径、每种模态插值
    traj_xy = torch.zeros(B, M, T, 2, device=device, dtype=dtype)
    for b in range(B):
        for m in range(M):
            tq = norm_disp[m]               # (T,)
            xp, yp = path[b, :, 0], path[b, :, 1]  # (P,)
            ref = norm_arc[b]               # (P,)
            x_interp = interp1d_linear(tq, ref, xp)
            y_interp = interp1d_linear(tq, ref, yp)
            traj_xy[b, m] = torch.stack([x_interp, y_interp], dim=-1)

    return traj_xy                          # (B, M, T, 2)

def speed_to_trajectory_vec(
        path: torch.Tensor,               # (B, P, 2)
        speed: torch.Tensor,              # (M, T)
        dt: float = 0.5) -> torch.Tensor:
    """
    完全并行版，去掉 for m in range(M)
    return: (B, M, T, 2)
    """
    B, P, _ = path.shape
    M, T = speed.shape
    device = path.device
    dtype = path.dtype

    # 1. 累计弧长  (B, P)
    arc = path_to_arc_length(path)

    # 2. 累计位移  (M, T)
    disp = speed * dt
    cum_disp = torch.cumsum(disp, dim=1)          # (M, T)

    # 3. 准备插值
    # 把查询点变成 (B*M*T,)  把参考轴变成 (B*M, P) 以便利用批量 searchsorted
    ref = arc.unsqueeze(1).expand(B, M, P).contiguous().view(B * M, P)  # (B*M, P)
    query = cum_disp.unsqueeze(0).expand(B, M, T).contiguous().view(B * M, T)  # (B*M, T)

    # 4. 批量 searchsorted
    idx = torch.searchsorted(ref, query, right=False)           # (B*M, T)
    idx = idx.clamp(1, P - 1)

    # 5. 批量 lerp
    x_all = path[..., 0].unsqueeze(1).expand(B, M, P).contiguous().view(B * M, P)  # (B*M, P)
    y_all = path[..., 1].unsqueeze(1).expand(B, M, P).contiguous().view(B * M, P)

    x0, x1 = x_all.gather(1, idx - 1), x_all.gather(1, idx)
    y0, y1 = y_all.gather(1, idx - 1), y_all.gather(1, idx)
    r0, r1 = ref.gather(1, idx - 1), ref.gather(1, idx)

    w = (query - r0) / (r1 - r0 + 1e-8)
    x_interp = x0 + w * (x1 - x0)
    y_interp = y0 + w * (y1 - y0)

    # 6.  reshape 回 (B, M, T, 2)
    traj_xy = torch.stack([x_interp, y_interp], dim=-1).view(B, M, T, 2)
    return traj_xy

def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision


@BBOX_CODERS.register_module()
class LatLonDecoder(object):
    def __init__(
        self,
        plan_config=dict(),
        use_lat_query=True,
        use_lon_query=True,
        use_rescore=False,
        time_points=None,
        score_thresh=0.5,
    ):
        super(LatLonDecoder, self).__init__()
        self.plan_config = plan_config
        self.use_lat_query = use_lat_query
        self.use_lon_query = use_lon_query
        self.use_rescore = use_rescore
        self.time_points = time_points
        self.score_thresh = score_thresh

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        if len(planning_output["planning_refine_results"]) > 0:
            planning_results = planning_output["planning_refine_results"][-1]
        else:
            planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        bs_indices = torch.arange(bs, device=period.device)

        lat_cls = planning_results["spatial_cls"].sigmoid().squeeze(-1)
        lat_reg = planning_results["spatial_reg"]
        lat_mode_idx = lat_cls.argmax(dim=-1)

        lon_cls = planning_results["vel_seq_cls"].sigmoid().squeeze(-1)
        lon_reg = planning_results["vel_seq_reg"]
        lon_mode_idx = lon_cls.argmax(dim=-1)

        if self.use_rescore:
            lat_anchor = lat_reg[0]
            lon_speed = lon_reg[0]
            time_points = lon_speed.new_tensor(self.time_points)
            time_interval = time_points[:, 1] - time_points[:, 0]
            lon_dist = (lon_speed * time_interval).cumsum(dim=-1)
            traj_anchor, traj_anchor_mask = interp_anchor_to_traj(lat_anchor, lon_dist)
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.score_thresh)
            ## rescore lat
            # for b in range(bs):
            #     col = col_mask[b].any(dim=1)
            #     lat_cls[b] = torch.where(
            #         col,
            #         0,
            #         lat_cls[b],
            #     )

            ## rescore lon
            for b in range(bs):
                lon_cls[b] = torch.where(
                    col_mask[b],
                    0,
                    lon_cls[b],
                )

            lat_mode_idx = lat_cls.argmax(dim=-1)
            lon_mode_idx = lon_cls.argmax(dim=-1)

        if not self.use_lat_query:
            lat_cls = lat_cls[bs_indices, lon_mode_idx]
            lat_mode_idx = lat_cls.argmax(dim=-1)

        if not self.use_lon_query:
            lon_cls = lon_cls[bs_indices, lat_mode_idx]
            lon_mode_idx = lon_cls.argmax(dim=-1)

        for b in range(bs):
            output[b]["spatial_cls"] = lat_cls[b].cpu()
            output[b]["spatial_reg"] = lat_reg[b].cpu()
            output[b]["spatial_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()

            output[b]["vel_seq_cls"] = lon_cls[b].cpu()
            output[b]["vel_seq_reg"] = lon_reg[b].cpu()
            output[b]["vel_seq_final"] = lon_reg[b, lon_mode_idx[b]].cpu()

        return output


@BBOX_CODERS.register_module()
class LatLonDecoderV6_1(object): ## traj decode
    def __init__(
        self,
        plan_config=dict(),
        use_lat_query=True,
        use_lon_query=True,
        use_rescore=False,
        decode_mode="traj",
    ):
        super(LatLonDecoderV6_1, self).__init__()
        self.plan_config = plan_config
        self.use_lat_query = use_lat_query
        self.use_lon_query = use_lon_query
        self.use_rescore = use_rescore
        self.decode_mode = decode_mode

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        if len(planning_output["planning_refine_results"]) > 0:
            planning_results = planning_output["planning_refine_results"][-1]
        else:
            planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        bs_indices = torch.arange(bs, device=period.device)
        lat_cls = planning_results["spatial_cls"].sigmoid().squeeze(-1)
        lat_mode_idx = lat_cls.argmax(dim=-1)
        lat_reg = planning_results["spatial_reg"]
        lon_reg = planning_results["vel_seq_reg"]

        for b in range(bs):
            if self.decode_mode == "traj":
                traj_cls = planning_results["vel_seq_cls"][b].sigmoid()
                max_flat = traj_cls.argmax()
                row, col = divmod(max_flat.item(), traj_cls.size(1))

                output[b]["spatial_cls"] = lat_cls[b].cpu()
                output[b]["spatial_reg"] = lat_reg[b].cpu()
                output[b]["spatial_reg_final"] = lat_reg[b, row].cpu()

                output[b]["vel_seq_cls"] = traj_cls[row].cpu()
                output[b]["vel_seq_reg"] = lon_reg[b].cpu()
                output[b]["vel_seq_final"] = lon_reg[b, col].cpu()
            elif self.decode_mode == "lat_lon":
                output[b]["spatial_cls"] = lat_cls[b].cpu()
                output[b]["spatial_reg"] = lat_reg[b].cpu()
                output[b]["spatial_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()

                lon_cls = planning_results["vel_seq_cls"][b, lat_mode_idx[b]].sigmoid()
                lon_mode_idx = lon_cls.argmax()
                output[b]["vel_seq_cls"] = lon_cls.cpu()
                output[b]["vel_seq_reg"] = lon_reg[b].cpu()
                output[b]["vel_seq_final"] = lon_reg[b, lon_mode_idx].cpu()

        return output


@BBOX_CODERS.register_module()
class LatLonDecoderSeqPara(object):
    def __init__(
        self,
        plan_config=dict(),
        use_lat_query=True,
        use_lon_query=True,
        use_rescore=False,
        lon_key="vel_seq_cls",
    ):
        super(LatLonDecoderSeqPara, self).__init__()
        self.plan_config = plan_config
        self.use_lat_query = use_lat_query
        self.use_lon_query = use_lon_query
        self.use_rescore = use_rescore
        self.lon_key = lon_key

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        if len(planning_output["planning_refine_results"]) > 0:
            planning_results = planning_output["planning_refine_results"][-1]
        else:
            planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        bs_indices = torch.arange(bs, device=period.device)
        lat_cls = planning_results["spatial_cls"].sigmoid().squeeze(-1)
        lat_reg = planning_results["spatial_reg"]
        lat_mode_idx = lat_cls.argmax(dim=-1)

        lon_cls = planning_results["vel_seq_cls"].sigmoid().squeeze(-1)
        lon_cls = lon_cls[bs_indices, lat_mode_idx]
        lon_reg = planning_results["vel_seq_reg"]
        lon_mode_idx = lon_cls.argmax(dim=-1)

        lon_aux_cls = planning_results["vel_seq_aux_cls"].sigmoid().squeeze(-1)
        lon_aux_mode_idx = lon_aux_cls.argmax(dim=-1)

        for b in range(bs):
            output[b]["spatial_cls"] = lat_cls[b].cpu()
            output[b]["spatial_reg"] = lat_reg[b].cpu()
            output[b]["spatial_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()

            if self.lon_key == "vel_seq_cls":
                output[b]["vel_seq_cls"] = lon_cls[b].cpu()
                output[b]["vel_seq_reg"] = lon_reg[b].cpu()
                output[b]["vel_seq_final"] = lon_reg[b, lon_mode_idx[b]].cpu()
            elif self.lon_key == "vel_seq_aux_cls":
                output[b]["vel_seq_cls"] = lon_aux_cls[b].cpu()
                output[b]["vel_seq_reg"] = lon_reg[b].cpu()
                output[b]["vel_seq_final"] = lon_reg[b, lon_aux_mode_idx[b]].cpu()

        return output

@BBOX_CODERS.register_module()
class LatLonTrajDecoder(object):
    def __init__(
        self,
        plan_config=dict(),
        lat_key="lat",
        lon_key="lon",
        lat_first=True,
        rescore=False,
        rescore_thresh=0.5,
    ):
        super(LatLonTrajDecoder, self).__init__()
        self.plan_config = plan_config
        self.lat_key = lat_key
        self.lon_key = lon_key
        self.lat_first = lat_first
        self.rescore = rescore
        self.rescore_thresh = rescore_thresh

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        if self.lat_key == "lat" and self.lon_key == "lon":
            output = self.decode_v1(det_output, motion_output, planning_results, output)
        if self.lat_key == "traj" and self.lon_key == "traj":
            output = self.decode_v2(det_output, motion_output, planning_results, output)
        if self.lat_key == "lat" and self.lon_key == "traj":
            output = self.decode_v3(det_output, motion_output, planning_results, output)
        if self.lat_key == "traj_f" and self.lon_key == "traj_f":
            output = self.decode_v4(det_output, motion_output, planning_results, output)
        return output


    def decode_v1(self, det_output, motion_output, planning_results, output):
        lat_cls = planning_results["lat_cls"].sigmoid().squeeze(-1)
        lat_reg = planning_results["lat_reg"]
        lat_mode_idx = lat_cls.argmax(dim=-1)
        bs = lat_cls.shape[0]

        lon_cls = planning_results["lon_cls"].sigmoid().squeeze(-1)
        lon_reg = planning_results["lon_reg"]
        lon_mode_idx = lon_cls.argmax(dim=-1)

        if self.rescore:
            traj_anchor = planning_results["traj_reg"][0]
            traj_anchor_mask = planning_results["traj_reg_mask"][0]
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.rescore_thresh)
            if self.lat_first:
                for b in range(bs):
                    lon_cls[b] = torch.where(
                        col_mask[b][lat_mode_idx[b]],
                        0,
                        lon_cls[b],
                    )
                lon_mode_idx = lon_cls.argmax(dim=-1)
            else:
                for b in range(bs):
                    lat_cls[b] = torch.where(
                        col_mask[b][:, lon_mode_idx[b]],
                        0,
                        lat_cls[b],
                    )
                lat_mode_idx = lat_cls.argmax(dim=-1)

        for b in range(bs):
            output[b]["lat_cls"] = lat_cls[b].cpu()
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()

            output[b]["lon_cls"] = lon_cls[b].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, lon_mode_idx[b]].cpu()

        return output


    def decode_v2(self, det_output, motion_output, planning_results, output):
        lat_reg = planning_results["lat_reg"]
        lon_reg = planning_results["lon_reg"]

        traj_cls = planning_results["traj_cls"].sigmoid().flatten(1, 2).squeeze(-1)
        traj_reg = planning_results["traj_reg"]
        traj_mode_idx = traj_cls.argmax(dim=-1)
        bs, num_lat_mode, num_lon_mode = traj_reg.shape[:3]
        if self.rescore:
            traj_anchor = planning_results["traj_reg"][0]
            traj_anchor_mask = planning_results["traj_reg_mask"][0]
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.rescore_thresh)
            for b in range(bs):
                mask = col_mask[b].flatten()
                traj_cls[b] = torch.where(
                    mask,
                    0,
                    traj_cls[b],
                )
            traj_mode_idx = traj_cls.argmax(dim=-1)

        for b in range(bs):
            row, col = divmod(traj_mode_idx[b].item(), num_lon_mode)
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, row].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, col].cpu()

        return output

    def decode_v3(self, det_output, motion_output, planning_results, output):
        lat_cls = planning_results["lat_cls"].sigmoid().squeeze(-1)
        lat_mode_idx = lat_cls.argmax(dim=-1)
        lat_reg = planning_results["lat_reg"]
        lon_reg = planning_results["lon_reg"]

        traj_cls = planning_results["traj_cls"].sigmoid().flatten(1, 2).squeeze(-1)
        traj_reg = planning_results["traj_reg"]
        traj_mode_idx = traj_cls.argmax(dim=-1)
        bs, num_lat_mode, num_lon_mode = traj_reg.shape[:3]

        if self.rescore:
            traj_anchor = planning_results["traj_reg"][0]
            traj_anchor_mask = planning_results["traj_reg_mask"][0]
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.rescore_thresh)
            if self.lat_first:
                for b in range(bs):
                    lon_cls[b] = torch.where(
                        col_mask[b][lat_mode_idx[b]],
                        0,
                        lon_cls[b],
                    )
                lon_mode_idx = lon_cls.argmax(dim=-1)
            else:
                for b in range(bs):
                    lat_cls[b] = torch.where(
                        col_mask[b][:, lon_mode_idx[b]],
                        0,
                        lat_cls[b],
                    )
                lat_mode_idx = lat_cls.argmax(dim=-1)

        for b in range(bs):
            row, col = divmod(traj_mode_idx[b].item(), num_lon_mode)
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, col].cpu()

        return output


    def decode_v4(self, det_output, motion_output, planning_results, output):
        traj_cls = planning_results["traj_cls"].sigmoid().flatten(1, 2).squeeze(-1)
        traj_reg = planning_results["traj_reg"].flatten(1, 2)
        traj_mode_idx = traj_cls.argmax(dim=-1)
        bs = traj_cls.shape[0]

        for b in range(bs):
            output[b]["traj_reg"] = traj_reg[b].cpu()
            output[b]["traj_reg_final"] = traj_reg[b, traj_mode_idx[b]].cpu()

        return output


@BBOX_CODERS.register_module()
class TemporalTrajDecoder(object):
    def __init__(
        self,
        plan_config=dict(),
        rescore=False,
        rescore_thresh=0.5,
    ):
        super(TemporalTrajDecoder, self).__init__()
        self.plan_config = plan_config
        self.rescore = rescore
        self.rescore_thresh = rescore_thresh

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        traj_cls = planning_results["traj_cls"].sigmoid().squeeze(-1)
        traj_reg = planning_results["traj_reg"]
        traj_mode_idx = traj_cls.argmax(dim=-1)

        for b in range(bs):
            output[b]["traj_reg"] = traj_reg[b].cpu()
            output[b]["traj_reg_final"] = traj_reg[b, traj_mode_idx[b]].cpu()

        return output

############################333
@BBOX_CODERS.register_module()
class V13Decoder(object):
    def __init__(
        self,
        plan_config=dict(),
        rescore=False,
        rescore_thresh=0.5,
    ):
        super(V13Decoder, self).__init__()
        self.plan_config = plan_config
        self.rescore = rescore
        self.rescore_thresh = rescore_thresh

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()

        lat_cls = planning_results["lat_cls"].sigmoid()
        lat_reg = planning_results["lat_anchor"]
        lat_mode_idx = lat_cls.argmax(dim=-1)
        lon_cls = planning_results["lon_cls"].sigmoid()
        lon_reg = planning_results["lon_anchor"]
        lon_mode_idx = lon_cls.argmax(dim=-1)
        traj_cls = planning_results["traj_cls"].sigmoid()
        traj_reg = planning_results["traj_reg"]
        traj_mode_idx = traj_cls.argmax(dim=-1)

        for b in range(bs):
            output[b]["lat_cls"] = lat_cls[b].cpu()
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()

            output[b]["lon_cls"] = lon_cls[b].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, lon_mode_idx[b]].cpu()

            output[b]["traj_reg"] = traj_reg[b].cpu()
            output[b]["traj_reg_final"] = traj_reg[b, traj_mode_idx[b]].cpu()

        return output


@BBOX_CODERS.register_module()
class V13LatLonTrajDecoder(object):
    def __init__(
        self,
        plan_config=dict(),
        lat_key="lat",
        lon_key="lon",
        lat_first=True,
        rescore=True,
        rescore_thresh=0.5,
        rescore_step=4,
        col_rescore=False,
        col_thresh=0.5,
    ):
        super(V13LatLonTrajDecoder, self).__init__()
        self.plan_config = plan_config
        self.lat_key = lat_key
        self.lon_key = lon_key
        self.lat_first = lat_first
        self.rescore = rescore
        self.rescore_thresh = rescore_thresh
        self.rescore_step = rescore_step
        self.col_rescore = col_rescore
        self.col_thresh = col_thresh

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        planning_results = planning_output["planning_results"][-1]
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        bs = period.shape[0]
        output = [{}] * bs

        for b in range(bs):
            output[b]["ego_period"] = period[b].cpu(),
            output[b]["ego_anchor_queue"] = decode_box(anchor_queue[b]).cpu()
        if self.lat_key == "lat" and self.lon_key == "lon":
            output = self.decode_v1(det_output, motion_output, planning_results, output)
        if self.lat_key == "traj" and self.lon_key == "traj":
            output = self.decode_v2(det_output, motion_output, planning_results, output)
        if self.lat_key == "lat" and self.lon_key == "traj":
            output = self.decode_v3(det_output, motion_output, planning_results, output)
        if self.lat_key == "traj_f" and self.lon_key == "traj_f":
            output = self.decode_v4(det_output, motion_output, planning_results, output)
        return output


    def decode_v1(self, det_output, motion_output, planning_results, output):
        lat_cls = planning_results["lat_cls"].sigmoid().squeeze(-1)
        lat_reg = planning_results["lat_reg"]
        lat_mode_idx = lat_cls.argmax(dim=-1)
        bs = lat_cls.shape[0]

        lon_cls = planning_results["lon_cls"].sigmoid().squeeze(-1)
        lon_reg = planning_results["lon_reg"]
        lon_mode_idx = lon_cls.argmax(dim=-1)

        if self.rescore:
            traj_anchor = planning_results["traj_reg"][0]
            traj_anchor_mask = planning_results["traj_reg_mask"][0]
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.rescore_thresh)
            if self.lat_first:
                for b in range(bs):
                    lon_cls[b] = torch.where(
                        col_mask[b][lat_mode_idx[b]],
                        0,
                        lon_cls[b],
                    )
                lon_mode_idx = lon_cls.argmax(dim=-1)
            else:
                for b in range(bs):
                    lat_cls[b] = torch.where(
                        col_mask[b][:, lon_mode_idx[b]],
                        0,
                        lat_cls[b],
                    )
                lat_mode_idx = lat_cls.argmax(dim=-1)

        for b in range(bs):
            output[b]["lat_cls"] = lat_cls[b].cpu()
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()

            output[b]["lon_cls"] = lon_cls[b].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, lon_mode_idx[b]].cpu()

        return output


    def decode_v2(self, det_output, motion_output, planning_results, output):
        lat_reg = planning_results["lat_anchor_filter"]
        lon_reg = planning_results["lon_anchor_filter"]

        traj_cls = planning_results["traj_cls"].sigmoid()
        traj_reg = planning_results["traj_reg"]
        traj_mode_idx = traj_cls.argmax(dim=-1)
        bs = traj_reg.shape[0]
        num_lat_mode = lat_reg.shape[1]
        num_lon_mode = lon_reg.shape[1]
        if self.col_rescore:
            col_mask = planning_results["collision_cls"].sigmoid() > self.col_thresh
            traj_cls = torch.where(
                col_mask,
                0,
                traj_cls,
            )
            traj_mode_idx = traj_cls.argmax(dim=-1)
        if self.rescore:
            traj_anchor = planning_results["traj_reg"][0].unflatten(0, (num_lat_mode, num_lon_mode))
            traj_anchor_mask = planning_results["traj_reg_mask"][0].unflatten(0, (num_lat_mode, num_lon_mode))
            traj_anchor_mask[:] = True
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.rescore_thresh)
            for b in range(bs):
                mask = col_mask[b][..., :self.rescore_step].any(dim=-1).any(dim=-1).flatten()
                traj_cls[b] = torch.where(
                    mask,
                    0,
                    traj_cls[b],
                )
            traj_mode_idx = traj_cls.argmax(dim=-1)

        for b in range(bs):
            row, col = divmod(traj_mode_idx[b].item(), num_lon_mode)
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, row].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, col].cpu()
            output[b]["traj_final"] = traj_reg[b, traj_mode_idx[b]].cpu()
            # Expose every candidate trajectory + its (post-rescore) score so the
            # downstream planner can rank and take the top-K. traj_cls here already
            # has collision/rescore-masked modes zeroed, so a descending sort by
            # score keeps the same winner as traj_final at rank 0.
            output[b]["traj_all"] = traj_reg[b].cpu()      # [N, T, 2]
            output[b]["traj_score"] = traj_cls[b].cpu()    # [N]

        return output

    def decode_v3(self, det_output, motion_output, planning_results, output):
        lat_cls = planning_results["lat_cls"].sigmoid().squeeze(-1)
        lat_mode_idx = lat_cls.argmax(dim=-1)
        lat_reg = planning_results["lat_reg"]
        lon_reg = planning_results["lon_reg"]

        traj_cls = planning_results["traj_cls"].sigmoid().flatten(1, 2).squeeze(-1)
        traj_reg = planning_results["traj_reg"]
        traj_mode_idx = traj_cls.argmax(dim=-1)
        bs, num_lat_mode, num_lon_mode = traj_reg.shape[:3]

        if self.rescore:
            traj_anchor = planning_results["traj_reg"][0]
            traj_anchor_mask = planning_results["traj_reg_mask"][0]
            col_mask = get_col_mask(traj_anchor, traj_anchor_mask, det_output, motion_output, self.rescore_thresh)
            if self.lat_first:
                for b in range(bs):
                    lon_cls[b] = torch.where(
                        col_mask[b][lat_mode_idx[b]],
                        0,
                        lon_cls[b],
                    )
                lon_mode_idx = lon_cls.argmax(dim=-1)
            else:
                for b in range(bs):
                    lat_cls[b] = torch.where(
                        col_mask[b][:, lon_mode_idx[b]],
                        0,
                        lat_cls[b],
                    )
                lat_mode_idx = lat_cls.argmax(dim=-1)

        for b in range(bs):
            row, col = divmod(traj_mode_idx[b].item(), num_lon_mode)
            output[b]["lat_reg"] = lat_reg[b].cpu()
            output[b]["lat_reg_final"] = lat_reg[b, lat_mode_idx[b]].cpu()
            output[b]["lon_reg"] = lon_reg[b].cpu()
            output[b]["lon_reg_final"] = lon_reg[b, col].cpu()

        return output


    def decode_v4(self, det_output, motion_output, planning_results, output):
        traj_cls = planning_results["traj_cls"].sigmoid().flatten(1, 2).squeeze(-1)
        traj_reg = planning_results["traj_reg"].flatten(1, 2)
        traj_mode_idx = traj_cls.argmax(dim=-1)
        bs = traj_cls.shape[0]

        for b in range(bs):
            output[b]["traj_reg"] = traj_reg[b].cpu()
            output[b]["traj_reg_final"] = traj_reg[b, traj_mode_idx[b]].cpu()

        return output
