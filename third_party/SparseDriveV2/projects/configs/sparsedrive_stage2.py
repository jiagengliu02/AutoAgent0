# SparseDriveV2 stage2: model and online input preprocessing only.
plugin = True

plugin_dir = "projects/mmdet3d_plugin/"

input_shape = (704, 384)

class_names = [
    'car',
    'van',
    'truck',
    'bicycle',
    'traffic_sign',
    'traffic_cone',
    'traffic_light',
    'pedestrian',
    'others',
]

map_class_names = [
    'Broken',
    'Solid',
    'SolidSolid',
    'Center',
    'TrafficLight',
    'StopSign',
]

num_classes = len(class_names)

num_map_classes = len(map_class_names)

roi_size = (30, 60)

num_sample = 20

fut_ts = 6

fut_mode = 6

embed_dims = 256

num_groups = 8

num_decoder = 6

num_single_frame_decoder = 1

num_single_frame_decoder_map = 1

use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed

strides = [4, 8, 16, 32]

num_levels = len(strides)

drop_out = 0.1

temporal = True

temporal_map = True

decouple_attn = True

decouple_attn_map = False

decouple_attn_motion = True

with_quality_estimation = True

queue_length = 21 # history + current

lat_fut_ts = 15

lat_fut_mode = 1024

lon_fut_ts = 6

lon_fut_mode = 256

time_points = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3)]

plan_config = dict(
    lat=dict(interval=1, num_pts=15, anchor="../../checkpoints/sparsedrivev2/anchors/path_1m_pts_15_1024_b2d_new_ego.npy", weight=1000/4),
    lon=dict(time_points=time_points, anchor="../../checkpoints/sparsedrivev2/anchors/vel_seq_K256_t30.npy", weight=100/2),
    traj=dict(time_points=time_points, anchor=f"../../checkpoints/sparsedrivev2/anchors/trajectory_{lat_fut_mode}_{lon_fut_mode}_b2d.npz", weight=50000/8/12),
    collision=dict(weight=1.0),
)

cond_config = dict(
    target_point=dict(),
)

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=True,
)

model = dict(
    type="SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    freeze_backbone=True,
    freeze_neck=True,
    freeze_perception=False,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained=None,
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    head=dict(
        type="SparseDriveHead",
        task_config=task_config,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor="../../checkpoints/sparsedrivev2/anchors/kmeans_det_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.9,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                # cls_allow_reverse=[class_names.index("barrier")],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        map_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor="../../checkpoints/sparsedrivev2/anchors/kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.9,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal_map
            else None,
            graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.25, -0.25, 0.5, -0.5),
                    ground_height=-1.84, # ground height in lidar frame
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[0.1] * 2,
                dn_trans_scale=[1.0] * 2,
                dn_rot_angle=2.5,
                dn_size_scale=0.05,
                max_dn_gt=16,
                noise_type=[1],
                dn_combination="all",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type='LinesL1Loss',
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(
                type="SparsePoint3DDecoder",
                score_threshold=0.5,
            ),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix='map',
        ),
        motion_plan_head=dict(
            type='MotionPlanningHeadV13',
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            lat_fut_ts=lat_fut_ts,
            lat_fut_mode=lat_fut_mode,
            lon_fut_ts=lon_fut_ts,
            lon_fut_mode=lon_fut_mode,
            filter_num=[(256, 64), (64, 16)],
            motion_anchor=f'../../checkpoints/sparsedrivev2/anchors/kmeans_motion_{fut_mode}.npy',
            plan_config=plan_config,
            embed_dims=embed_dims,
            decouple_attn=decouple_attn_motion,
            instance_queue=dict(
                type="InstanceQueue",
                embed_dims=embed_dims,
                queue_length=queue_length,
                tracking_threshold=0.2,
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
                ego_init_mode=1, # 0: front feature, 1: random embedding, 2: zero init.
                cond_config=cond_config,
            ),
            cond_encoder=dict(
                type="ConditionEncoder",
                cond_config=cond_config,
                embed_dims=embed_dims,
            ),
            operation_order=(
                [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "cross_gnn",
                    "norm",
                    "ffn",                    
                    "norm",
                    "motion_pred",
                ] * 2 +
                [
                    "get_plan_query",
                ] + [
                    ## lat attn
                    "lat_cond_cross_attn",
                    "lat_deformable",
                    "lat_agent_gnn",
                    "lat_map_gnn",
                    "lat_mode_norm",
                    "lat_mode_gnn",
                    "lat_mode_norm",
                    ## lon attn
                    "lon_agent_gnn",
                    "lon_map_gnn",
                    "lon_mode_norm",
                    "lon_mode_gnn",
                    "lon_mode_norm",
                    ## filter
                    "lat_lon_pred",
                    "traj_pred",
                    "decoder_change",
                ] + [
                    ## lat attn
                    "lat_cond_cross_attn",
                    "lat_deformable",
                    "lat_agent_gnn",
                    "lat_map_gnn",
                    "lat_mode_norm",
                    "lat_mode_gnn",
                    "lat_mode_norm",
                    ## lon attn
                    "lon_agent_gnn",
                    "lon_map_gnn",
                    "lon_mode_norm",
                    "lon_mode_gnn",
                    "lon_mode_norm",
                    ## filter
                    "lat_lon_pred",
                    "traj_deformable",
                    "traj_mode_gnn",
                    "traj_mode_norm",
                    "traj_pred",
                    "decoder_change",
                ]
            ),
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            graph_model=dict(
                type="CustomMultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            cross_graph_model=dict(
                type="CustomMultiheadAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            mode_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            cond_graph_model=dict(
                type="CustomMultiheadAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            mode_norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            motion_pred_layer=dict(
                type="MotionPredModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
            ),
            lat_lon_pred_layer=dict(
                type="LatLonPredModuleV13",
                embed_dims=embed_dims,
                plan_config=plan_config,
            ),
            traj_pred_layer=dict(
                type="TrajPredModule",
                embed_dims=embed_dims,
                plan_config=plan_config,
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.0,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="add",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=lat_fut_ts,
                    num_learnable_pts=0,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84, # ground height in lidar frame
                ),
            ),
            traj_deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.0,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="add",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=fut_ts,
                    num_learnable_pts=0,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84, # ground height in lidar frame
                ),
            ),
            motion_sampler=dict(
                type="MotionTarget",
            ),
            motion_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2
            ),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
            planning_sampler=dict(
                type="V13PlanningTarget",
                plan_config=plan_config,
            ),
            lat_plan_loss_cls=dict(type="CustomClassificationLoss", loss_type="IndiceFocalv2"),
            lon_plan_loss_cls=dict(type="CustomClassificationLoss", loss_type="IndiceFocalv2"),
            traj_plan_loss_cls=dict(type="CustomClassificationLoss", loss_type="IndiceFocalv2"),
            col_plan_loss_cls=dict(type="CustomClassificationLoss", loss_type="BCE"),
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),
            planning_decoder=dict(
                type="V13LatLonTrajDecoder",
                plan_config=plan_config,
                lat_key="traj",
                lon_key="traj",
                lat_first=True,
            ),
            num_det=50,
            num_map=30,
        ),
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            'ego_status',
            'gt_ego_fut_cmd',
            "tp_near",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}
