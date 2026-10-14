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
from ..blocks import linear_relu_ln



@PLUGIN_LAYERS.register_module()
class ConditionEncoder(BaseModule):
    def __init__(self, cond_config, embed_dims):
        super(ConditionEncoder, self).__init__()
        self.cond_config = cond_config
        if "target_point" in cond_config:
            self.tp_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 2),
                Linear(embed_dims, embed_dims),
            )
        if "target_point_far" in cond_config:
            self.tp_far_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 2),
                Linear(embed_dims, embed_dims),
            )
        if "ego_status" in cond_config:
            self.es_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 10),
                Linear(embed_dims, embed_dims),
            )
        if "ego_speed" in cond_config:
            self.speed_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 1),
                Linear(embed_dims, embed_dims),
            )
        if "route" in cond_config:
            self.route_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 20),
                Linear(embed_dims, embed_dims),
            )
            if "dropout_p" in cond_config["route"]:
                self.route_dropout_p = cond_config["route"]["dropout_p"]
            else:
                self.route_dropout_p = 0
            if "noise_scale" in cond_config["route"]:
                self.route_noise_scale = cond_config["route"]["noise_scale"]
            else:
                self.route_noise_scale = 0


    def forward(self, metas):
        conditions = []
        if "target_point" in self.cond_config:
            tp_embedding = self.tp_encoder(metas["tp_near"].float())
            conditions.append(tp_embedding)
        if "target_point_far" in self.cond_config:
            tp_embedding = self.tp_far_encoder(metas["tp_far"].float())
            conditions.append(tp_embedding)
        if "ego_status" in self.cond_config:
            es_embedding = self.es_encoder(metas["ego_status"].float())
            conditions.append(es_embedding)
        if "ego_speed" in self.cond_config:
            ego_speed = metas["ego_status"][:, 6:7]
            speed_embedding = self.speed_encoder(ego_speed.float())
            conditions.append(speed_embedding)
        if "route" in self.cond_config:
            route = metas["route"].float().flatten(-2, -1)
            if self.route_noise_scale > 0:
                noise = (torch.rand(route.shape, device=route.device) * 2 - 1) * self.route_noise_scale
                route += noise
            route_embedding = self.route_encoder(route)
            if self.route_dropout_p > 0 and self.training:
                mask = torch.rand(route_embedding.shape[0], device=route_embedding.device) >= self.route_dropout_p
                route_embedding *= mask.unsqueeze(1)
            conditions.append(route_embedding)

        output = torch.stack(conditions, dim=1)
        return output
