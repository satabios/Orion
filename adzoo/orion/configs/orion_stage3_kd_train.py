# Knowledge Distillation Configuration for ORION EVA-L Backbone Training
# Focus: Only train the student EVA-L backbone (8 heads) with KD from teacher EVA-L (16 heads)
# All other components (neck, heads, LLM) are frozen and loaded from pre-trained model

_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]
backbone_norm_cfg = dict(type='LN', requires_grad=True)

# Point cloud range and voxel configuration (same as original)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Map classes and configuration (same as original)
map_classes = ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign']
map_fixed_ptsnum_per_gt_line = 11
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
past_frames = 2
future_frames = 6
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2

ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# NameMapping configuration (same as original)
NameMapping = {
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    "vehicle.tesla.model3": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.mercedes.coupe": 'car',
    "vehicle.bmw.grandtourer": 'car',
    "vehicle.toyota.prius": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.audi.a2": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.nissan.patrol": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.mini.cooper_s": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.nissan.micra": 'car',
    "vehicle.tesla.cybertruck": 'car',
    "vehicle.seat.leon": 'car',
    "vehicle.mustang.mustang": 'car',
    "vehicle.jeep.wrangler_rubicon": 'car',
    "vehicle.bmw.isetta": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.citroen.c3": 'car',
    "vehicle.ford.ambulance": "van",
    "vehicle.carlamotors.firetruck": 'truck',
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',
    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    "static.prop.warningconstruction" : 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",
    "static.prop.constructioncone": 'traffic_cone',
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0002": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0006": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0008": 'pedestrian',
    "walker.pedestrian.0009": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0011": 'pedestrian',
    "walker.pedestrian.0012": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
}

class_names = ['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian']

# Additional variables required by ORION (same as original)
use_memory = True
use_gen_token = True
use_col_loss = True
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

# Training configuration with KD parameters
use_memory = True
num_gpus = 32
batch_size = 4
num_iters_per_epoch = 234769 // (num_gpus * batch_size)
num_epochs = 6
llm_path = None  # Disable LLM components for backbone-only training
use_gen_token = False  # Disable for backbone-only training
use_col_loss = True
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']
mix_qa_training = False  # Disable QA training for backbone-only KD

# Knowledge Distillation Configuration
kd_config = dict(
    teacher_backbone_path='ckpt/orion/orion.pth',  # Full model path (will extract backbone)
    distillation_alpha=0.7,  # Higher weight for backbone distillation (70% distillation, 30% task)
    distillation_temperature=3.0,  # Lower temperature for tighter teacher following
    feature_distill_weight=0.5,  # High weight for feature-level distillation
    attention_distill_weight=0.3,  # Attention pattern transfer
    output_distill_weight=0.2,  # Lower weight for output distillation
    warmup_epochs=1,  # Shorter warmup since only training backbone
    warmup_alpha=0.8,  # Very high distillation weight during warmup
    final_alpha=0.6,  # Still high distillation weight after warmup
    pruning_ratio=0.5,  # 50% head reduction (16 -> 8 heads)
    head_selection_strategy='magnitude',  # Head selection method
    freeze_non_backbone=True,  # Freeze everything except backbone
)

# Student Model Configuration (Backbone-only KD)
model = dict(
    type='OrionStudent',  # Use student model for backbone KD
    teacher_backbone_path=kd_config['teacher_backbone_path'],
    distillation_alpha=kd_config['distillation_alpha'],
    freeze_non_backbone=kd_config['freeze_non_backbone'],
    
    # Distillation loss configuration (focused on backbone features)
    distillation_loss=dict(
        type='CombinedKDLoss',
        feature_distill_weight=kd_config['feature_distill_weight'],
        attention_distill_weight=kd_config['attention_distill_weight'],
        output_distill_weight=kd_config['output_distill_weight'],
        vl_distill_weight=0.0,  # No VL distillation for backbone-only training
        temperature=kd_config['distillation_temperature']
    ),
    
    save_path='./results_planning_kd/',
    use_grid_mask=True,
    frozen=False,
    use_lora=False,  # Disable LoRA for backbone-only training
    tokenizer=None,  # Disable tokenizer for backbone-only training
    lm_head=None,    # Disable LM head for backbone-only training
    use_gen_token=use_gen_token,
    use_diff_decoder=False,
    use_col_loss=use_col_loss,
    loss_plan_reg=dict(type='L1Loss', loss_weight=3.0),
    loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=3.0, dis_thresh=1.0),
    loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=3.0),
    
    # Student EVA-ViT backbone with 50% fewer heads
    img_backbone=dict(
        type='EVAViTStudent',  # Use student backbone
        img_size=640, 
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=8,  # Reduced from 16 to 8 (50% pruning)
        teacher_num_heads=16,  # Original number of heads for weight transfer
        mlp_ratio=4*2/3,
        window_block_indexes=(
            list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + 
            list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + 
            list(range(18, 20)) + list(range(21, 23))
        ),
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True, 
        frozen=False,
    ), 
    
    pts_bbox_head=dict(
        type='OrionHead',
        num_classes=9,  # Should be 9 like original, not len(class_names)
        in_channels=1024,  # Should be 1024, not _dim_
        out_dims=4096,
        num_query=600,  # Should be 600, not 256
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,  # Missing from our config
        num_extra=256,
        n_control=11,
        match_with_velo=False,  # Missing from our config
        pred_traffic_light_state=True,  # Missing from our config
        use_col_loss=use_col_loss,
        use_memory=use_memory,  # Missing from our config
        scalar=10,  # Missing from our config
        noise_scale=1.0,  # Missing from our config
        dn_weight=1.0,  # Missing from our config
        split=0.75,  # Missing from our config
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=_dim_,
            output_dimension=_dim_,
            num_layers=6,
            embed_dims=_dim_,
            num_heads=8,
            feedforward_dims=_ffn_dim_,
            dropout=0.1,
            with_cp=True,
            flash_attn=True),
        # Additional transformer configurations required by OrionHead
        use_pe=False,
        motion_transformer_decoder=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False,
        ),
        memory_decoder_transformer=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False,
        ),
        # bbox_coder and other configurations
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=9
        ),
        score_threshold=0.2,
        class_agnostic_nms=dict(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], 
            compensate=[0, 0, 0.3, 0, 0, 0, 0, 0.3, 0],
            pre_max_size=1000,
            post_max_size=300,
            nms_thr=0.1,
        ),
        sync_cls_avg_factor=False,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    
    # Map head configuration (same as original)
    map_head=dict(
        type='OrionHeadM',
        num_classes=6,
        in_channels=1024,
        out_dims=4096,
        memory_len=600,
        with_mask=True,
        topk_proposals=300,
        num_lane=1800,
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=256,
        n_control=11,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0],
        score_threshold=0.2,
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
    ),
    
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))))

# Dataset configuration (same as original)
dataset_type = 'B2DOrionDataset'  # Correct dataset for ORION training
data_root = 'data/Chat-B2D/'
ann_file_train = 'data/Chat-B2D/chat_b2d_train_infos.pkl'
ann_file_val = 'data/Chat-B2D/chat_b2d_val_infos.pkl'
ann_file_test = 'data/Chat-B2D/chat_b2d_val_infos.pkl'
map_root = 'data/Chat-B2D/'
map_file = 'data/Chat-B2D/map_infos.pkl'

file_client_args = dict(backend='disk')

# Training pipeline (same as original)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True, file_client_args=file_client_args, img_root=data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='LoadTrajData'),
    dict(type='LoadMapData', map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs','input_ids','gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks','ego_fut_cmd', 'ego_lcf_feat','vlm_labels','can_bus', 'traffic_state_mask', 'traffic_state']+collect_keys),
]

# Test pipeline (same as original)
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='LoadTrajData'),
    dict(type='LoadMapData', map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs','input_ids','gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks','ego_fut_cmd', 'ego_lcf_feat','vlm_labels','can_bus', 'traffic_state_mask', 'traffic_state']+collect_keys),
]

# Data configuration
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        seq_mode=True,
        seq_split_num=1,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        modality=input_modality,
        queue_length=1,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        test_mode=False,
        # ... (rest of train config)
    ),
    val=dict(
        type=dataset_type,
        seq_mode=True,
        seq_split_num=1,
        pipeline=test_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        ann_file=ann_file_val,
        data_root=data_root,
        modality=input_modality,
        test_mode=True),
    test=dict(
        type=dataset_type,
        seq_mode=True,
        seq_split_num=1,
        pipeline=test_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        ann_file=ann_file_test,
        data_root=data_root,
        modality=input_modality,
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# Optimizer configuration (only backbone parameters will be updated)
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # Higher LR since only training backbone
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=1.0),  # Full LR for backbone (only trainable component)
            # All other components are frozen, so LR doesn't matter
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))  # Lower grad clip for stability

# Learning rate scheduler
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,  # Shorter warmup for backbone-only training
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-2)

# Training configuration (shorter since only training backbone)
runner = dict(type='IterBasedRunner', max_iters=3 * num_iters_per_epoch)  # Only 3 epochs needed
evaluation = dict(interval=num_iters_per_epoch, pipeline=test_pipeline)  # Evaluate every epoch
find_unused_parameters = False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)

# Logging
log_config = dict(
    interval=10, 
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook")
    ]
)

# Load configuration 
load_from = 'ckpt/orion/orion.pth'  # Load full pre-trained model (non-backbone components will be frozen)
resume_from = None

# Custom hooks for backbone-only KD training
custom_hooks = [
    dict(
        type='BackboneKDHook',
        teacher_backbone_path=kd_config['teacher_backbone_path'],
        priority='NORMAL'
    )
]

# Workflow
workflow = [('train', 1)]