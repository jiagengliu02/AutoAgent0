from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.datasets.pipelines.transform import interp_paths_speeds
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position, gen_sineembed_for_position_multi
from ..blocks import linear_relu_ln
from ..instance_bank import topk, topk_with_indice
from .motion_blocks import RouteEncoder, FlattenRouteEncoder
from .rescore_utils import interp_anchor_to_traj, interp_feature, interp_feature_lowmem

@HEADS.register_module()
class MotionPlanningHeadV13(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        lat_fut_ts=15,
        lat_fut_mode=1024,
        lon_fut_ts=1,
        lon_fut_mode=45,
        filter_num=None,
        motion_anchor=None,
        plan_config=None,
        plan_refine_config=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        cond_encoder=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        mode_graph_model=None,
        cond_graph_model=None,
        norm_layer=None,
        mode_norm_layer=None,
        ffn=None,
        motion_pred_layer=None,
        traj_pred_layer=None,
        lat_lon_pred_layer=None,
        plan_refine_layer=None,
        deformable_model=None,
        traj_deformable_model=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        planning_sampler_refine=None,
        lat_plan_loss_cls=None,
        lon_plan_loss_cls=None,
        traj_plan_loss_cls=None,
        col_plan_loss_cls=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=20,
        det_attn_conf=False,
        map_attn_conf=False,
        det_attn_dist=False,
        map_attn_dist=False,
        his_dropout_p=0.0,
        lon_deform=False,
        lon_cumsum=False,
        motion_emb=False,
        sel_num_lat=None,
        sel_num_lon=None,
        lat_topk=None,
        lon_topk=None,
        decoder_weight=[1,]*6,
    ):
        super(MotionPlanningHeadV13, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.lat_fut_ts = lat_fut_ts
        self.lat_fut_mode = lat_fut_mode
        self.lon_fut_ts = lon_fut_ts
        self.lon_fut_mode = lon_fut_mode
        self.filter_num = filter_num
        self.plan_config = plan_config
        self.plan_refine_config = plan_refine_config

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.cond_encoder = build(cond_encoder, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.planning_sampler_refine = build(planning_sampler_refine, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "lat_agent_gnn": [cross_graph_model, ATTENTION],
            "lat_map_gnn": [cross_graph_model, ATTENTION],
            "lon_agent_gnn": [cross_graph_model, ATTENTION],
            "lon_map_gnn": [cross_graph_model, ATTENTION],
            "traj_agent_gnn": [cross_graph_model, ATTENTION],
            "traj_map_gnn": [cross_graph_model, ATTENTION],
            "mode_gnn": [mode_graph_model, ATTENTION],
            "lat_mode_gnn": [mode_graph_model, ATTENTION],
            "lon_mode_gnn": [mode_graph_model, ATTENTION],
            "traj_mode_gnn": [mode_graph_model, ATTENTION],
            "lat_lon_mode_gnn": [mode_graph_model, ATTENTION],
            "motion_gnn": [graph_model, ATTENTION],
            "cond_cross_attn": [cond_graph_model, ATTENTION],
            "lat_cond_cross_attn": [cond_graph_model, ATTENTION],
            "lon_cond_cross_attn": [cond_graph_model, ATTENTION],
            "traj_cond_cross_attn": [cond_graph_model, ATTENTION],
            "deformable": [deformable_model, ATTENTION],
            "lat_deformable": [deformable_model, ATTENTION],
            "traj_deformable": [traj_deformable_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "mode_norm": [mode_norm_layer, NORM_LAYERS],
            "lat_mode_norm": [mode_norm_layer, NORM_LAYERS],
            "lon_mode_norm": [mode_norm_layer, NORM_LAYERS],
            "traj_mode_norm": [mode_norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "motion_pred": [motion_pred_layer, PLUGIN_LAYERS],
            "traj_pred": [traj_pred_layer, PLUGIN_LAYERS],
            "lat_lon_pred": [lat_lon_pred_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.lat_plan_loss_cls = build_loss(lat_plan_loss_cls)
        self.lon_plan_loss_cls = build_loss(lon_plan_loss_cls)
        if traj_plan_loss_cls is not None:
            self.traj_plan_loss_cls = build_loss(traj_plan_loss_cls)
        self.col_plan_loss_cls = build_loss(col_plan_loss_cls)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        lat_anchor = np.load(plan_config["lat"]["anchor"])
        self.lat_anchor = nn.Parameter(
            torch.tensor(lat_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        lon_anchor = np.load(plan_config["lon"]["anchor"])
        self.lon_anchor = nn.Parameter(
            torch.tensor(lon_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        trajectory_data = np.load(plan_config["traj"]["anchor"])
        self.traj_anchor = nn.Parameter(
            torch.from_numpy(trajectory_data["trajectory"]).float(),
            requires_grad=False
        )
        self.traj_anchor_mask = nn.Parameter(
            torch.from_numpy(trajectory_data["trajectory_mask"]).float(),
            requires_grad=False
        )

        # plan mode query
        self.lat_plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, lat_fut_ts * 2),
            Linear(embed_dims, embed_dims),
        )
        self.lon_plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, lon_fut_ts),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map

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

    def get_motion_anchor(
        self,
        classification,
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected, det_confidence_selected, det_anchors_selected), det_indices = topk_with_indice(
            det_confidence, self.num_det, instance_feature, anchor_embed, det_confidence, det_anchors
        )

        map_instance_feature = map_output["instance_feature"]
        num_map_anchor = map_instance_feature.shape[1]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected, map_confidence_selected, map_anchors_selected), map_indices = topk_with_indice(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed, map_confidence, map_anchors
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        device = instance_feature.device
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)

        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== motion init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))

        # =========== plan init ===========

        lat_anchor = torch.tile(self.lat_anchor, (bs, 1 ,1, 1)) # bs, num_lat_mode, num_pts, 2
        lon_anchor = torch.tile(self.lon_anchor, (bs, 1 ,1)) # bs, num_lon_mode, num_vel
        traj_anchor = torch.tile(self.traj_anchor, (bs, 1 ,1, 1, 1)) # bs, num_lat_mode, num_lon_mode, num_vel, 2
        traj_anchor_mask = torch.tile(self.traj_anchor_mask, (bs, 1 ,1, 1))

        lat_mode_query = self.lat_plan_anchor_encoder(lat_anchor.flatten(-2))
        lon_mode_query = self.lon_plan_anchor_encoder(lon_anchor)
        lat_anchor_embed = ego_anchor_embed.repeat(1, lat_anchor.shape[1], 1)
        lon_anchor_embed = ego_anchor_embed.repeat(1, lon_anchor.shape[1], 1)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =========== encode condition ===========
        cond_embedding = self.cond_encoder(metas)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_results = []
        decoder_idx = 0
        plan_result = {}
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "motion_pred":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                motion_cls, motion_reg = self.layers[i](motion_query)
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
            elif op == "get_plan_query":
                ego_feature = instance_feature[:, -1]
                ego_anchor_embed = anchor_embed[:, -1]
                lat_mode_query = lat_mode_query + (ego_feature + ego_anchor_embed).unsqueeze(1)
                lon_mode_query = lon_mode_query + (ego_feature + ego_anchor_embed).unsqueeze(1)
            elif op == "lat_mode_gnn" or op == "lat_mode_norm":
                lat_mode_query = self.layers[i](lat_mode_query)
            elif op == "lon_mode_gnn" or op == "lon_mode_norm":
                lon_mode_query = self.layers[i](lon_mode_query)
            elif op == "lat_cond_cross_attn":
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    key=cond_embedding,
                )
            elif op == "lat_agent_gnn":
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    instance_feature_selected,
                    instance_feature_selected,
                    # query_pos=lat_anchor_embed,
                    # key_pos=anchor_embed_selected,
                )
            elif op == "lat_map_gnn":
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    # query_pos=lat_anchor_embed,
                    # key_pos=map_anchor_embed_selected,
                )
            elif op == "lon_cond_cross_attn":
                lon_mode_query = self.layers[i](
                    lon_mode_query,
                    key=cond_embedding,
                )
            elif op == "lon_agent_gnn":
                lon_mode_query = self.layers[i](
                    lon_mode_query,
                    instance_feature_selected,
                    instance_feature_selected,
                    # query_pos=lon_anchor_embed,
                    # key_pos=anchor_embed_selected,
                )
            elif op == "lon_map_gnn":
                lon_mode_query = self.layers[i](
                    lon_mode_query,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    # query_pos=lon_anchor_embed,
                    # key_pos=map_anchor_embed_selected,
                )
            elif op == "lat_deformable":
                lat_anchor_flat = lat_anchor.flatten(-2)
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    lat_anchor_flat,
                    None,
                    feature_maps,
                    metas,
                    None,
                )
            elif op == "lat_lon_pred":
                plan_result["lat_anchor"] = lat_anchor.clone()
                plan_result["lon_anchor"] = lon_anchor.clone()
                (
                    lat_mode_query,
                    lon_mode_query,
                    lat_anchor,
                    lon_anchor,
                    lat_cls,
                    lon_cls,
                    traj_mode_query,
                    traj_anchor,
                    traj_anchor_mask,
                ) = self.layers[i](
                    lat_mode_query,
                    lon_mode_query,
                    lat_anchor,
                    lon_anchor,
                    traj_anchor,
                    traj_anchor_mask,
                    self.filter_num[decoder_idx],
                )
                traj_point_query = None
                plan_result["lat_cls"] = lat_cls.clone()
                plan_result["lon_cls"] = lon_cls.clone()
            elif op == "traj_mode_gnn" or op == "traj_mode_norm":
                traj_mode_query = self.layers[i](traj_mode_query)
            elif op == "traj_cond_cross_attn":
                traj_mode_query = self.layers[i](
                    traj_mode_query,
                    key=cond_embedding,
                )
            elif op == "traj_agent_gnn":
                traj_mode_query = self.layers[i](
                    traj_mode_query,
                    instance_feature_selected,
                    instance_feature_selected,
                )
            elif op == "traj_map_gnn":
                traj_mode_query = self.layers[i](
                    traj_mode_query,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                )
            elif op == "traj_deformable":
                traj_anchor_flat = traj_anchor.flatten(1, 2).flatten(-2)
                traj_mode_query, traj_point_query = self.layers[i](
                    traj_mode_query,
                    traj_anchor_flat,
                    None,
                    feature_maps,
                    metas,
                    None,
                    return_kps_features=True,
                )
            elif op == "traj_pred":
                traj_output = self.layers[i](
                    traj_mode_query,
                    traj_point_query,
                    traj_anchor,
                    traj_anchor_mask,
                )
                plan_result.update(traj_output)
            elif op == "decoder_change":
                planning_results.append(plan_result)
                plan_result = {}
                decoder_idx += 1

        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        status = plan_result.get("ego_status_reg")
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_results[-1]["lat_anchor_filter"] = lat_anchor.clone()
        planning_results[-1]["lon_anchor_filter"] = lon_anchor.clone()
        planning_output = {
            "planning_results": planning_results,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output

    def loss(self,
        motion_model_outs,
        planning_model_outs,
        data,
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target,
                cls_weight,
                reg_pred,
                reg_target,
                reg_weight,
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        planning_results = model_outs["planning_results"]
        output = {}
        for decoder_idx, planning_result in enumerate(planning_results):
            (
                lat_cls,
                lat_cls_target,
                lat_cls_weight,
                lon_cls,
                lon_cls_target,
                lon_cls_weight,
                traj_cls,
                traj_cls_target,
                traj_cls_weight,
                col_cls,
                col_label,
                point_col_cls,
                point_col_label,
            ) = self.planning_sampler.sample(
                planning_result,
                data,
            )
            lat_cls_loss = self.lat_plan_loss_cls(lat_cls, lat_cls_target, lat_cls_weight) * self.plan_config["lat"]["weight"]
            output[f"lat_cls_loss_{decoder_idx}"] = lat_cls_loss
            lon_cls_loss = self.lon_plan_loss_cls(lon_cls, lon_cls_target, lon_cls_weight) * self.plan_config["lon"]["weight"]
            output[f"lon_cls_loss_{decoder_idx}"] = lon_cls_loss
            if traj_cls is not None:
                traj_cls_loss = self.traj_plan_loss_cls(traj_cls, traj_cls_target, traj_cls_weight) * self.plan_config["traj"]["weight"]
                output[f"traj_cls_loss_{decoder_idx}"] = traj_cls_loss
            if col_cls is not None:
                col_cls_loss = self.col_plan_loss_cls(col_cls, col_label)
                output[f"col_cls_loss_{decoder_idx}"] = col_cls_loss * self.plan_config["collision"]["weight"]
            if point_col_cls is not None:
                point_col_loss = self.col_plan_loss_cls(point_col_cls, point_col_label)
                output[f"point_col_cls_loss_{decoder_idx}"] = point_col_loss * self.plan_config["point_collision"]["weight"]

        return output


    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output,
            data,
        )
        return motion_result, planning_result
