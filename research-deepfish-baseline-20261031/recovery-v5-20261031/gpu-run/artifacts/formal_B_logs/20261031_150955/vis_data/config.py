auto_scale_lr = dict(base_batch_size=1, enable=False)
default_hooks = dict(
    checkpoint=None,
    logger=dict(interval=10, type='LoggerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/content/deepfish_ab_dd122a9217/payload/assets/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=[
            1,
            2,
            3,
        ],
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=False, log_scale=0.0, max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    language_model=dict(
        add_pooling_layer=False,
        name='/content/deepfish_ab_dd122a9217/payload/assets/bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = []
randomness = dict(deterministic=True, diff_rank_seed=False, seed=20261031)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/dev64.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/content/deepfish_ab_dd122a9217/payload/data/',
        metainfo=dict(classes=('fish', )),
        pipeline=[
            dict(
                backend_args=None,
                color_type='color',
                imdecode_backend='cv2',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                1333,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=[
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                ],
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/content/deepfish_ab_dd122a9217/payload/data/annotations/dev64.json',
    format_only=False,
    metric='bbox',
    prefix='dev',
    type='CocoMetric')
train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=99999)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='annotations/support5.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/content/deepfish_ab_dd122a9217/payload/data/',
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(classes=('fish', )),
        pipeline=[
            dict(
                transforms=[
                    dict(
                        backend_args=None,
                        color_type='color',
                        imdecode_backend='cv2',
                        type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        level=0,
                        max_mag=0.8,
                        min_mag=0.8,
                        prob=1.0,
                        type='Color'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(
                        transforms=[
                            [
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            480,
                                            1333,
                                        ),
                                        (
                                            512,
                                            1333,
                                        ),
                                        (
                                            544,
                                            1333,
                                        ),
                                        (
                                            576,
                                            1333,
                                        ),
                                        (
                                            608,
                                            1333,
                                        ),
                                        (
                                            640,
                                            1333,
                                        ),
                                        (
                                            672,
                                            1333,
                                        ),
                                        (
                                            704,
                                            1333,
                                        ),
                                        (
                                            736,
                                            1333,
                                        ),
                                        (
                                            768,
                                            1333,
                                        ),
                                        (
                                            800,
                                            1333,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                            ],
                            [
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            400,
                                            4200,
                                        ),
                                        (
                                            500,
                                            4200,
                                        ),
                                        (
                                            600,
                                            4200,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                                dict(
                                    allow_negative_crop=True,
                                    crop_size=(
                                        384,
                                        600,
                                    ),
                                    crop_type='absolute_range',
                                    type='RandomCrop'),
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            480,
                                            1333,
                                        ),
                                        (
                                            512,
                                            1333,
                                        ),
                                        (
                                            544,
                                            1333,
                                        ),
                                        (
                                            576,
                                            1333,
                                        ),
                                        (
                                            608,
                                            1333,
                                        ),
                                        (
                                            640,
                                            1333,
                                        ),
                                        (
                                            672,
                                            1333,
                                        ),
                                        (
                                            704,
                                            1333,
                                        ),
                                        (
                                            736,
                                            1333,
                                        ),
                                        (
                                            768,
                                            1333,
                                        ),
                                        (
                                            800,
                                            1333,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                            ],
                        ],
                        type='RandomChoice'),
                    dict(
                        meta_keys=[
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                            'flip',
                            'flip_direction',
                            'text',
                            'custom_entities',
                        ],
                        type='PackDetInputs'),
                ],
                type='AuditedPipeline'),
        ],
        return_classes=True,
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(round_up=False, shuffle=True, type='DefaultSampler'))
val_cfg = None
val_dataloader = None
val_evaluator = None
work_dir = '/content/deepfish_ab_dd122a9217/artifacts/formal_B_logs'
