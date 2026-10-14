import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_cmd=3,
        with_path=False,
    ):
        super(MotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.with_path = with_path

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )
        if self.with_path:
            self.path_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 40),
                )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, self.num_cmd * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        if self.with_path:
            path = self.path_branch(ego_feature + ego_anchor_embed)
        else:
            path = None
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status, path


@PLUGIN_LAYERS.register_module()
class PlanRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(PlanRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,
    ):
        bs = plan_query.shape[0]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        return plan_cls, plan_reg


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModuleV2(BaseModule): ## no reg
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_cmd=3,
        with_path=False,
    ):
        super(MotionPlanningRefinementModuleV2, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.with_path = with_path

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )
        if self.with_path:
            self.path_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 40),
                )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        if self.with_path:
            path = self.path_branch(ego_feature + ego_anchor_embed)
        else:
            path = None
        return motion_cls, motion_reg, plan_cls, planning_status, path


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModuleV3(BaseModule): ## add pred col
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_cmd=3,
        with_path=False,
    ):
        super(MotionPlanningRefinementModuleV3, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.with_path = with_path

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_col_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )
        if self.with_path:
            self.path_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 40),
                )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_col_cls = self.plan_col_cls_branch(plan_query).squeeze(-1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        if self.with_path:
            path = self.path_branch(ego_feature + ego_anchor_embed)
        else:
            path = None
        return motion_cls, motion_reg, plan_cls, plan_col_cls, planning_status, path

@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModuleV4(BaseModule): ## add pred driveable
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_cmd=3,
        with_path=False,
    ):
        super(MotionPlanningRefinementModuleV4, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.with_path = with_path

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_col_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_road_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )
        if self.with_path:
            self.path_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 40),
                )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_col_cls = self.plan_col_cls_branch(plan_query).squeeze(-1)
        plan_road_cls = self.plan_road_cls_branch(plan_query).squeeze(-1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        if self.with_path:
            path = self.path_branch(ego_feature + ego_anchor_embed)
        else:
            path = None
        return motion_cls, motion_reg, plan_cls, plan_col_cls, plan_road_cls, planning_status, path

class FlattenRouteEncoder(nn.Module):
    def __init__(self, in_dim=20, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, route):
        route = route.flatten(1, 2)
        h = self.mlp(route)
        return h

class RouteEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden=64, route_dropout=False, dropout_p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.route_dropout = route_dropout
        if self.route_dropout:
            self.dropout = nn.Dropout1d(p=dropout_p)   # route-level dropout

    def forward(self, route):
        # route: (bs, 10, 2)
        h = self.mlp(route)            # (bs, 10, hidden)
        if self.route_dropout:
            h = self.dropout(h)            # 整条 10×hidden 一起随机置 0
        h = h.mean(dim=1)              # (bs, hidden)
        return h

@PLUGIN_LAYERS.register_module()
class PlanOffsetRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(PlanOffsetRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,
        plan_anchor,
    ):
        bs = plan_query.shape[0]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg_offset = self.plan_reg_branch(plan_query).reshape(bs, self.ego_fut_mode, self.ego_fut_ts, 2)
        plan_anchor = plan_anchor + plan_reg_offset
        return plan_anchor, plan_cls

@PLUGIN_LAYERS.register_module()
class PlanClsRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(PlanClsRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,
        plan_anchor,
    ):
        bs = plan_query.shape[0]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        return plan_anchor, plan_cls

@PLUGIN_LAYERS.register_module()
class CroaseToFineBlock(BaseModule):
    def __init__(
        self,
        filter_num=128,
        filter_mode=0,
        graph_model=None,
        norm_layer=None,
        ffn=None,
        deformable_model=None,
        refine_layer=None,
        operation_order=None,
    ):
        super(CroaseToFineBlock, self).__init__()
        self.filter_num = filter_num
        self.filter_mode = filter_mode

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.operation_order = operation_order
        self.op_config_map = {
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "deformable": [deformable_model, ATTENTION],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(self, plan_anchor, plan_cls, plan_query, metas, feature_maps):
        plan_cls = plan_cls[:, 0]
        plan_query = plan_query[:, 0]

        bs = plan_cls.size(0)
        bs_idx = torch.arange(bs, device=plan_cls.device).view(-1, 1)  # (bs, 1)

        if self.filter_mode == 1 and self.training:
            gt_traj = metas["gt_ego_fut_trajs"].cumsum(dim=-2)
            mask = metas["gt_ego_fut_masks"]
            dist = gt_traj.unsqueeze(1) - plan_anchor
            dist = torch.linalg.norm(dist, dim=-1)
            dist = dist * mask.unsqueeze(1)
            dist = dist.mean(dim=-1)
            _, anchor_idx = torch.topk(-dist, k=self.filter_num, dim=1)
        else:
            _, anchor_idx = torch.topk(plan_cls, k=self.filter_num, dim=1)

        filtered_anchor = plan_anchor[bs_idx, anchor_idx]
        filtered_query = plan_query[bs_idx, anchor_idx]

        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "deformable":
                filtered_query = self.layers[i](
                    filtered_query,
                    filtered_anchor.flatten(-2, -1),
                    None,
                    feature_maps,
                    metas,
                    None,
                )
            elif op == "gnn":
                filtered_query = self.layers[i](filtered_query)
            elif op == "norm" or op == "ffn":
                filtered_query = self.layers[i](filtered_query)
            elif op == "refine":
                plan_anchor, plan_cls = self.layers[i](filtered_query, filtered_anchor)

        return plan_anchor, plan_cls.unsqueeze(1)


@PLUGIN_LAYERS.register_module()
class MotionPredModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
    ):
        super(MotionPredModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        return motion_cls, motion_reg


@PLUGIN_LAYERS.register_module()
class PlanningPredModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
    ):
        super(PlanningPredModule, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        self.branches = nn.ModuleDict()
        for key, value in plan_config.items():
            for pred_type, pred_dim in zip(value["pred_types"], value["pred_dims"]):
                if pred_type == "cls":
                    in_loops = 1
                    out_loops = 2
                elif pred_type == "reg":
                    in_loops = 2
                    out_loops = 2
                name = key + "_" + pred_type + "_" + "branch"
                self.branches[name] = nn.Sequential(
                    *linear_relu_ln(embed_dims, in_loops, out_loops),
                    Linear(embed_dims, pred_dim),
                )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        for key, branch in self.branches.items():
            if "cls" in key:
                nn.init.constant_(branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,
        plan_anchor,
        ego_feature,
        ego_anchor_embed,
        lon_plan_query=None,
    ):
        plan_result = {}
        for key, value in self.plan_config.items():
            for pred_type, pred_dim in zip(value["pred_types"], value["pred_dims"]):
                out_name = key + "_" + pred_type 
                branch_name = out_name + "_" + "branch"
                if value.get("ego_pred") == True: 
                    output = self.branches[branch_name](ego_feature + ego_anchor_embed)
                elif value.get("lon_pred") == True: 
                    output = self.branches[branch_name](lon_plan_query).squeeze(-1)
                else:
                    output = self.branches[branch_name](plan_query)
                if ("spatial" in key or "temporal" in key):
                    if pred_type == "reg":
                        if plan_anchor.get(key) is not None:
                            anchor = plan_anchor[key]
                            output = anchor + output.unflatten(-1, (-1, 2))
                            plan_anchor[key] = output
                        else:
                            output = output.unflatten(-1, (-1, 2)).cumsum(-2)
                            plan_anchor[key] = output

                    elif pred_type == "cls": 
                        output = output.squeeze(-1)
                        if "reg" not in value["pred_types"]:
                            plan_result[key + "_" + "reg"] = plan_anchor[key]

                plan_result[out_name] = output

        return plan_result, plan_anchor


@PLUGIN_LAYERS.register_module()
class LatLonPredModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
        pred_col=False,
    ):
        super(LatLonPredModule, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        lat_pred_dim = plan_config["spatial"]["pred_dim"]
        self.lat_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lat_pred_dim),
        )
        lon_pred_dim = plan_config["vel_seq"]["pred_dim"]
        self.lon_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lon_pred_dim),
        )

        self.pred_col = pred_col
        if self.pred_col:
            self.col_cls_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, lon_pred_dim),
            )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.lat_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.lon_cls_branch[-1].bias, bias_init)
        if self.pred_col:
            nn.init.constant_(self.lon_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        lat_mode_query,
        lon_mode_query,
        lat_anchor,
        lon_anchor,
    ):
        output = {}
        if lat_mode_query is not None:
            lat_cls = self.lat_cls_branch(lat_mode_query)
        else:
            lat_cls = self.lat_cls_branch(lon_mode_query)
        output["spatial_cls"] = lat_cls
        output["spatial_reg"] = lat_anchor

        if lon_mode_query is not None:
            lon_cls = self.lon_cls_branch(lon_mode_query)
        else:
            lon_cls = self.lon_cls_branch(lat_mode_query)
        output["vel_seq_cls"] = lon_cls
        output["vel_seq_reg"] = lon_anchor

        if self.pred_col:
            col_cls = self.col_cls_branch(lat_mode_query)
            output["col_cls"] = col_cls

        return output


@PLUGIN_LAYERS.register_module()
class LatLonPredModuleV9_0(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
    ):
        super(LatLonPredModuleV9_0, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        lat_pred_dim = plan_config["spatial"]["pred_dim"]
        self.lat_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lat_pred_dim),
        )
        lon_pred_dim = plan_config["vel_seq"]["pred_dim"]
        self.lon_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lon_pred_dim),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.lat_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.lon_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        lat_mode_query,
        lon_mode_query,
        lat_anchor,
        lon_anchor,
    ):
        output = {}
        lat_cls = self.lat_cls_branch(lat_mode_query)
        output["spatial_cls"] = lat_cls
        output["spatial_reg"] = lat_anchor

        lat_lon_query = lat_mode_query.unsqueeze(2) + lon_mode_query.unsqueeze(1)
        lon_cls = self.lon_cls_branch(lat_lon_query)
        output["vel_seq_cls"] = lon_cls
        output["vel_seq_reg"] = lon_anchor

        lon_cls = self.lon_cls_branch(lon_mode_query)
        output["vel_seq_aux_cls"] = lon_cls
        output["vel_seq_aux_reg"] = lon_anchor
        return output

@PLUGIN_LAYERS.register_module()
class LatLonPredModuleV9_1(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
    ):
        super(LatLonPredModuleV9_1, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        lat_pred_dim = plan_config["spatial"]["pred_dim"]
        self.lat_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lat_pred_dim),
        )
        lon_pred_dim = plan_config["vel_seq"]["pred_dim"]
        self.lon_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lon_pred_dim),
        )
        self.lon_aux_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, lon_pred_dim),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.lat_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.lon_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.lon_aux_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        lat_mode_query,
        lon_mode_query,
        lat_anchor,
        lon_anchor,
    ):
        output = {}
        lat_cls = self.lat_cls_branch(lat_mode_query)
        output["spatial_cls"] = lat_cls
        output["spatial_reg"] = lat_anchor

        lat_lon_query = lat_mode_query.unsqueeze(2) + lon_mode_query.unsqueeze(1)
        lon_cls = self.lon_cls_branch(lat_lon_query)
        output["vel_seq_cls"] = lon_cls
        output["vel_seq_reg"] = lon_anchor

        lon_cls = self.lon_aux_cls_branch(lon_mode_query)
        output["vel_seq_aux_cls"] = lon_cls
        output["vel_seq_aux_reg"] = lon_anchor
        return output

@PLUGIN_LAYERS.register_module()
class LatLonTrajPredModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
    ):
        super(LatLonTrajPredModule, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        self.lat_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )
        self.lon_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )
        self.traj_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )
        if "collision" in self.plan_config:
            self.collision_cls_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, 1),
            )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.lat_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.lon_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.traj_cls_branch[-1].bias, bias_init)
        if "collision" in self.plan_config:
            nn.init.constant_(self.collision_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        lat_mode_query,
        lon_mode_query,
        traj_query,
        lat_anchor,
        lon_anchor,
        traj_anchor,
        traj_anchor_mask,
    ):
        output = {}
        lat_cls = self.lat_cls_branch(lat_mode_query)
        output["lat_cls"] = lat_cls
        output["lat_reg"] = lat_anchor

        lon_cls = self.lon_cls_branch(lon_mode_query)
        output["lon_cls"] = lon_cls
        output["lon_reg"] = lon_anchor

        traj_cls = self.traj_cls_branch(traj_query)
        output["traj_cls"] = traj_cls
        output["traj_reg"] = traj_anchor
        output["traj_reg_mask"] = traj_anchor_mask

        if "collision" in self.plan_config:
            collision_cls = self.collision_cls_branch(traj_query)
            output["collision_cls"] = collision_cls

        return output

@PLUGIN_LAYERS.register_module()
class TemporalTrajPredModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
    ):
        super(TemporalTrajPredModule, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        self.traj_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )
        if "gt_col" in self.plan_config:
            self.gt_col_cls_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, 1),
            )
        if "pred_col" in self.plan_config:
            self.pred_col_cls_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, 1),
            )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.traj_cls_branch[-1].bias, bias_init)
        if "gt_col" in self.plan_config:
            nn.init.constant_(self.gt_col_cls_branch[-1].bias, bias_init)
        if "pred_col" in self.plan_config:
            nn.init.constant_(self.pred_col_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_query,
        traj_anchor,
    ):
        output = {}
        traj_cls = self.traj_cls_branch(traj_query)
        output["traj_cls"] = traj_cls
        output["traj_reg"] = traj_anchor

        if "gt_col" in self.plan_config:
            gt_col_cls = self.gt_col_cls_branch(traj_query)
            output["gt_col_cls"] = gt_col_cls

        if "pred_col" in self.plan_config:
            pred_col_cls = self.gt_col_cls_branch(traj_query)
            output["pred_col_cls"] = pred_col_cls

        return output

##################################################3
@PLUGIN_LAYERS.register_module()
class LatLonPredModuleV13(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
        filter_mode="score",
    ):
        super(LatLonPredModuleV13, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        self.filter_mode = filter_mode
        self.lat_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )
        self.lon_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.lat_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.lon_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        path_embed,
        vel_embed,
        path_vocab,
        vel_vocab,
        traj_vocab,
        traj_mask,
        filter_num,
    ):
        num_path = path_embed.shape[1]
        num_vel = vel_embed.shape[1]

        path_scores = self.lat_cls_branch(path_embed).squeeze(-1)
        vel_scores = self.lon_cls_branch(vel_embed).squeeze(-1)

        filter_traj_vocab = traj_vocab.clone()
        filter_traj_mask = traj_mask.clone()

        if num_path > filter_num[0]:
            if self.training and self.filter_mode == "gt":
                target_path = targets["path"][:, :self.config.n_pts]
                target_path_mask = targets["path_mask"]
                dist = (path_vocab - target_path[:, None]) ** 2
                dist = dist.sum((-2, -1)) * self.config.path_sigmas[self.decoder_idx]
                topk_path_scores, topk_path_indices = torch.topk(-dist, self.config.path_filter_num[self.decoder_idx], dim=1)
            else:
                topk_path_scores, topk_path_indices = torch.topk(path_scores, filter_num[0], dim=1)
            filter_path_embed = torch.gather(path_embed, 1, topk_path_indices.unsqueeze(-1).expand(-1, -1, path_embed.shape[-1]))
            filter_path_vocab = torch.gather(path_vocab, 1, topk_path_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, path_vocab.shape[-2], path_vocab.shape[-1]))
            filter_traj_vocab = torch.gather(filter_traj_vocab, 1, topk_path_indices[:, :, None, None, None].expand(-1, -1, filter_traj_vocab.shape[-3], filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]))
            filter_traj_mask = torch.gather(filter_traj_mask, 1, topk_path_indices[:, :, None, None].expand(-1, -1, filter_traj_mask.shape[-2], filter_traj_mask.shape[-1]))
        else:
            filter_path_embed = path_embed
            filter_path_vocab = path_vocab

        if num_vel > filter_num[1]:
            if self.training and self.filter_mode == "gt":
                target_vel = targets["velocity"]
                dist = (vel_vocab - target_vel[:, None]).abs()
                dist = dist.sum(-1) * self.velocity_sigmas[self.decoder_idx]
                topk_vel_scores, topk_vel_indices  = torch.topk(-dist, filter_num[1], dim=1)
            else:
                topk_vel_scores, topk_vel_indices = torch.topk(vel_scores, filter_num[1], dim=1)
            filter_vel_embed = torch.gather(vel_embed, 1, topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_embed.shape[-1]))
            filter_vel_vocab = torch.gather(vel_vocab, 1, topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_vocab.shape[-1]))
            filter_traj_vocab = torch.gather(filter_traj_vocab, 2, topk_vel_indices[:, None, :, None, None].expand(-1, filter_traj_vocab.shape[-4], -1, filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]))
            filter_traj_mask = torch.gather(filter_traj_mask, 2, topk_vel_indices[:, None, :, None].expand(-1, filter_traj_mask.shape[-3], -1, filter_traj_mask.shape[-1]))
        else:
            filter_vel_embed = vel_embed
            filter_vel_vocab = vel_vocab

        traj_embed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
        traj_embed = traj_embed.flatten(1, 2)

        return (
            filter_path_embed,
            filter_vel_embed,
            filter_path_vocab,
            filter_vel_vocab,
            path_scores,
            vel_scores,
            traj_embed,
            filter_traj_vocab,
            filter_traj_mask,
        )


@PLUGIN_LAYERS.register_module()
class TrajPredModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        plan_config=None,
    ):
        super(TrajPredModule, self).__init__()
        self.embed_dims = embed_dims
        self.plan_config = plan_config
        self.traj_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, 1),
        )
        if "collision" in self.plan_config:
            self.collision_cls_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, 1),
            )
        if "point_collision" in self.plan_config:
            self.point_collision_cls_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, 1),
            )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.02)
        nn.init.constant_(self.traj_cls_branch[-1].bias, bias_init)
        if "collision" in self.plan_config:
            nn.init.constant_(self.collision_cls_branch[-1].bias, bias_init)
        if "point_collision" in self.plan_config:
            nn.init.constant_(self.point_collision_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_query,
        traj_point_query,
        traj_anchor,
        traj_anchor_mask,
    ):
        output = {}
        traj_cls = self.traj_cls_branch(traj_query).squeeze(-1)
        output["traj_cls"] = traj_cls
        output["traj_reg"] = traj_anchor.flatten(1, 2)
        output["traj_reg_mask"] = traj_anchor_mask.flatten(1, 2)

        if "collision" in self.plan_config:
            collision_cls = self.collision_cls_branch(traj_query).squeeze(-1)
            output["collision_cls"] = collision_cls

        if "point_collision" in self.plan_config:
            point_collision_cls = self.point_collision_cls_branch(traj_point_query).squeeze(-1)
            output["point_collision_cls"] = point_collision_cls

        return output
