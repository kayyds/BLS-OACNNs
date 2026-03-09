# 阶段1：配置优化试验 - 几何感知恢复
# 目标：通过配置调整恢复几何感知能力

_base_ = ["../_base_/default_runtime.py"]

# 基础配置优化
batch_size = 12           # 介于BLS(16)和OACNNs(12)之间
mix_prob = 0.8
empty_cache = False
enable_amp = True
sync_bn = True
enable_wandb = False
# 模型配置 - 阶段1优化参数
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="BLS_OACNNs",
        in_channels=9,
        num_classes=20,
        embed_channels=64,
        enc_channels=[64, 64, 128, 256],
        groups=[4, 4, 8, 16],

        # 关键改进1：调整编码器深度，保留几何感知
        enc_depth=[3, 3, 7, 6],  # 原BLS [2,3,6,4] -> [3,3,7,6]
                                   # Stage1增加1层，Stage3增加1层

        # 关键改进2：降低BLS比例，给几何感知更多空间
        bls_ratio=0.4,           # 原BLS 0.7 -> 0.4，减少BLS使用

        # 关键改进3：提升解码器通道，补偿可能的细节损失
        dec_channels=[128, 128, 256, 256],  # 原BLS [96,96,128,256]

        # 保持原有配置
        point_grid_size=[[8, 12, 16, 16], [6, 9, 12, 12],
                        [4, 6, 8, 8], [3, 4, 6, 6]],
        dec_depth=[2, 2, 2, 2],
        enc_num_ref=[16, 16, 16, 16],
        use_bls=True,
        incremental_learning=True,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# 训练配置优化
epoch = 600                     # 原BLS 600 -> 650，给几何感知更多训练时间
optimizer = dict(
    type="AdamW",
    lr=0.0015,                  # 原BLS 0.002 -> 0.0015，平衡速度和稳定性
    weight_decay=0.02
)
# 学习率调度
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = '/root/autodl-tmp/pointcept_data/scannet_processed'

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "bookshelf", "picture", "counter", "desk",
        "curtain", "refridgerator", "shower curtain", "toilet", "sink",
        "bathtub", "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.15, dropout_application_ratio=0.2),  # 减少dropout
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                # 减少TTA数量，因为BLS本身更稳定
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
            ],
        ),
    ),
)

# 阶段1试验参数记录
experiment_config = {
    "stage": "config_optimization",
    "key_changes": [
        "enc_depth: [2,3,6,4] -> [3,3,7,6]",
        "bls_ratio: 0.7 -> 0.4",
        "dec_channels: [96,96,128,256] -> [128,128,256,256]",
        "batch_size: 16 -> 14",
        "epoch: 600 -> 650",
        "lr: 0.002 -> 0.0015"
    ],
    "expected_improvement": "+0.5% to +1.0% mIoU",
    "training_time_estimate": "7-8 hours"
}