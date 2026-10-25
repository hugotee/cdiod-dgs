_data_root = 'data/CDIOD/RUOD/'
ann_file_path = 'valid/annotations_without_background.json'
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'caption_prompt',
        ),
        type='PackDetInputs'),
]
caption_prompt = None
custom_hooks = [
    dict(
        cfg=[
            dict(type='moe_lora'),
            dict(type='moe_group_init'),
        ],
        type='WeightsTransformHook'),
    dict(cfg=dict(type='ema'), type='MergeHook'),
    dict(type='DomainPredictorHooK'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.DGS.dgs',
    ])
data_metric = 'CDIODMetric'
data_root = 'data/CDIOD/'
dataset = dict(
    ann_file='valid/annotations_without_background.json',
    caption_prompt=None,
    data_prefix=dict(img='valid/'),
    data_root='data/CDIOD/RUOD/',
    metainfo=(
        'DIOR',
        'PascalVOC',
        'RUOD',
    ),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            1333,
            800,
        ), type='Resize'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'text',
                'custom_entities',
                'caption_prompt',
            ),
            type='PackDetInputs'),
    ],
    return_classes=True,
    test_mode=True,
    type='CDIOD_Agnostic_Dataset')
dataset_prefixes = [
    'DIOR',
    'PascalVOC',
    'RUOD',
]
dataset_type = 'CDIOD_Agnostic_Dataset'
datasets = [
    dict(
        ann_file='valid/annotations_without_background.json',
        caption_prompt=None,
        data_prefix=dict(img='valid/'),
        data_root='data/CDIOD/DIOR/',
        metainfo=(
            'DIOR',
            'PascalVOC',
            'RUOD',
        ),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                    'caption_prompt',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CDIOD_Agnostic_Dataset'),
    dict(
        ann_file='valid/annotations_without_background.json',
        caption_prompt=None,
        data_prefix=dict(img='valid/'),
        data_root='data/CDIOD/PascalVOC/',
        metainfo=(
            'DIOR',
            'PascalVOC',
            'RUOD',
        ),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                    'caption_prompt',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CDIOD_Agnostic_Dataset'),
    dict(
        ann_file='valid/annotations_without_background.json',
        caption_prompt=None,
        data_prefix=dict(img='valid/'),
        data_root='data/CDIOD/RUOD/',
        metainfo=(
            'DIOR',
            'PascalVOC',
            'RUOD',
        ),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                    'caption_prompt',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CDIOD_Agnostic_Dataset'),
]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        save_optimizer=True,
        save_param_scheduler=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='GroundingVisualizationHook'))
default_scope = 'mmdet'
diagnostic_protocol = dict(
    accumulation=2,
    cap=None,
    effective_batch=4,
    feature_distillation_configuration=
    'inter-intra retained; implementation checks inter-class',
    fixed_data_and_augmentation_seed=True,
    full_validation_prompt_classes=20,
    gpu_ids=[
        3,
    ],
    kind='VOC_single_domain_AB',
    microbatch=2,
    primary_metric='normal routing AP',
    seed=1498185232)
distn_cfg = dict(
    feat_distn=dict(
        img_loss=dict(loss_weight=3.0, reduction='mean', type='L2Loss'),
        subtype='opt1',
        text_loss=dict(loss_weight=5.0, reduction='mean', type='L2Loss'),
        type='inter-intra'),
    future_class=False,
    label_distn=dict(
        label_iou_th=0.7, mode='hardlabel', sigma=0.4,
        type='threshold_pseudo'),
    query_distn=dict(
        num_aux_query=900, num_matching_query=900, type='seperate_queryinit'),
    type='distillation')
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
lang_model_name = 'weights/bert-base-uncased'
load_from = '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/prefix/PascalVOC_Task4-3/epoch_12.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
metainfo = (
    'DIOR',
    'PascalVOC',
    'RUOD',
)
metric = dict(
    ann_file='data/CDIOD/RUOD/valid/annotations_without_background.json',
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
        'AR@100',
        'AR@300',
    ],
    type='CDIODMetric',
    use_mp_eval=False)
metrics = [
    dict(
        ann_file='data/CDIOD/DIOR/valid/annotations_without_background.json',
        metric='bbox',
        metric_items=[
            'mAP',
            'mAP_50',
            'mAP_75',
            'mAP_s',
            'mAP_m',
            'mAP_l',
            'AR@100',
            'AR@300',
        ],
        type='CDIODMetric',
        use_mp_eval=False),
    dict(
        ann_file=
        'data/CDIOD/PascalVOC/valid/annotations_without_background.json',
        metric='bbox',
        metric_items=[
            'mAP',
            'mAP_50',
            'mAP_75',
            'mAP_s',
            'mAP_m',
            'mAP_l',
            'AR@100',
            'AR@300',
        ],
        type='CDIODMetric',
        use_mp_eval=False),
    dict(
        ann_file='data/CDIOD/RUOD/valid/annotations_without_background.json',
        metric='bbox',
        metric_items=[
            'mAP',
            'mAP_50',
            'mAP_75',
            'mAP_s',
            'mAP_m',
            'mAP_l',
            'AR@100',
            'AR@300',
        ],
        type='CDIODMetric',
        use_mp_eval=False),
]
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
        frozen_stages=-1,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    bbox_head=dict(
        contrastive_cfg=dict(bias=None, log_scale=None, max_text_len=256),
        distn_cfg=dict(
            feat_distn=dict(
                img_loss=dict(
                    loss_weight=3.0, reduction='mean', type='L2Loss'),
                subtype='opt1',
                text_loss=dict(
                    loss_weight=5.0, reduction='mean', type='L2Loss'),
                type='inter-intra'),
            future_class=False,
            label_distn=dict(
                label_iou_th=0.7,
                mode='hardlabel',
                sigma=0.4,
                type='threshold_pseudo'),
            query_distn=dict(
                num_aux_query=900,
                num_matching_query=900,
                type='seperate_queryinit'),
            type='distillation'),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        setting='cur_text',
        sync_cls_avg_factor=True,
        trunc_class=[
            0,
            256,
        ],
        type='GroundingDINOHead_inc_DGS'),
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
    distn_cfg=dict(
        feat_distn=dict(
            img_loss=dict(loss_weight=3.0, reduction='mean', type='L2Loss'),
            subtype='opt1',
            text_loss=dict(loss_weight=5.0, reduction='mean', type='L2Loss'),
            type='inter-intra'),
        future_class=False,
        label_distn=dict(
            label_iou_th=0.7,
            mode='hardlabel',
            sigma=0.4,
            type='threshold_pseudo'),
        query_distn=dict(
            num_aux_query=900,
            num_matching_query=900,
            type='seperate_queryinit'),
        type='distillation'),
    dn_cfg=None,
    domain_predictor_cfg=dict(
        expand_th=150,
        feat_path='UNUSED_CACHED_STATISTICS_REQUIRED',
        grouping_trace_path=
        '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/branches/A_original/grouping_trace.jsonl',
        max_group_kl=None,
        min_eig_ratio=0.001,
        multilevel=False,
        ood_th=500,
        stats_path=
        '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/frozen_stats',
        task_id_mapping_path=
        '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/branches/A_original/task_id_mapping.yaml',
        type='svd'),
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
        num_cp=0,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    frozen_cfg=dict(
        backbone_frozen=True,
        decoder_frozen=True,
        encoder_frozen=True,
        exclude_keywords=[
            'lora_',
        ],
        head_frozen=True,
        language_model_frozen=True,
        neck_frozen=True),
    language_model=dict(
        add_pooling_layer=False,
        name='/home/data/dq/cdiod-dgs/weights/bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    moe_cfg=dict(
        alpha=32.0,
        dropout=0.0,
        experts_num=1,
        group_cfg=dict(
            lambda_A=0.2, lambda_B=0.2, merge_method='ema', type='rep'),
        r=16,
        replace_dec_layer_ids=[],
        replace_enc_layer_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
        replace_layer_type=[
            'enc_ffn_img',
            'enc_ffn_text',
        ],
        top_k=1,
        type='moe_adaptive_expand_lora'),
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
    num_tasks=4,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    seen_tasks=[
        'PascalVOC_Task4-1',
        'PascalVOC_Task4-2',
        'PascalVOC_Task4-3',
        'PascalVOC_Task4-4',
    ],
    task_id=3,
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO_DGS',
    vis_cfg=dict(save_path='', type='none'),
    with_box_refine=True)
moe_cfg = dict(
    alpha=32.0,
    dropout=0.0,
    experts_num=1,
    group_cfg=dict(lambda_A=0.2, lambda_B=0.2, merge_method='ema', type='rep'),
    r=16,
    replace_dec_layer_ids=[],
    replace_enc_layer_ids=[
        0,
        1,
        2,
        3,
        4,
        5,
    ],
    replace_layer_type=[
        'enc_ffn_img',
        'enc_ffn_text',
    ],
    top_k=1,
    type='moe_adaptive_expand_lora')
optim_wrapper = dict(
    accumulative_counts=2,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.1))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            11,
        ],
        type='MultiStepLR'),
]
prefix = 'RUOD'
randomness = dict(deterministic=False, diff_rank_seed=False, seed=1498185232)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='valid/annotations_without_background.json',
                caption_prompt=None,
                data_prefix=dict(img='valid/'),
                data_root='data/CDIOD/PascalVOC/',
                metainfo=('PascalVOC', ),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1333,
                        800,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                            'text',
                            'custom_entities',
                            'caption_prompt',
                        ),
                        type='PackDetInputs'),
                ],
                return_classes=True,
                test_mode=True,
                type='CDIOD_Agnostic_Dataset'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dataset_prefixes=[
        'PascalVOC',
    ],
    evaluation_tasks=
    'PascalVOC_Task4-1,PascalVOC_Task4-2,PascalVOC_Task4-3,PascalVOC_Task4-4',
    metrics=[
        dict(
            ann_file=
            'data/CDIOD/PascalVOC/valid/annotations_without_background.json',
            classwise=True,
            metric='bbox',
            metric_items=[
                'mAP',
                'mAP_50',
                'mAP_75',
                'mAP_s',
                'mAP_m',
                'mAP_l',
                'AR@100',
                'AR@300',
            ],
            outfile_prefix=
            '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/branches/A_original/validation_predictions',
            type='CDIODMetric',
            use_mp_eval=False),
    ],
    type='CDIODMultiDatasetsEvaluator')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        backend='pillow',
        keep_ratio=True,
        scale=(
            800,
            1333,
        ),
        type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'ori_text',
            'custom_entities',
            'caption_prompt',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=12)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='train/annotations_without_background_Task4-4.json',
        data_prefix=dict(img='train/'),
        data_root='data/CDIOD/PascalVOC/',
        distn_cfg=dict(
            feat_distn=dict(
                img_loss=dict(
                    loss_weight=3.0, reduction='mean', type='L2Loss'),
                subtype='opt1',
                text_loss=dict(
                    loss_weight=5.0, reduction='mean', type='L2Loss'),
                type='inter-intra'),
            future_class=False,
            label_distn=dict(
                label_iou_th=0.7,
                mode='hardlabel',
                sigma=0.4,
                type='threshold_pseudo'),
            query_distn=dict(
                num_aux_query=900,
                num_matching_query=900,
                type='seperate_queryinit'),
            task_id_mapping_path=
            '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/branches/A_original/task_id_mapping.yaml',
            type='distillation'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=
        'PascalVOC_Task4-1,PascalVOC_Task4-2,PascalVOC_Task4-3,PascalVOC_Task4-4',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
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
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'ori_text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        seen_tasks=
        'PascalVOC_Task4-1,PascalVOC_Task4-2,PascalVOC_Task4-3,PascalVOC_Task4-4',
        type='CDIOD_Agnostic_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'ori_text',
            'custom_entities',
        ),
        type='PackDetInputs'),
]
use_mp_eval = False
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='valid/annotations_without_background.json',
                caption_prompt=None,
                data_prefix=dict(img='valid/'),
                data_root='data/CDIOD/PascalVOC/',
                metainfo=('PascalVOC', ),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1333,
                        800,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                            'text',
                            'custom_entities',
                            'caption_prompt',
                        ),
                        type='PackDetInputs'),
                ],
                return_classes=True,
                test_mode=True,
                type='CDIOD_Agnostic_Dataset'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'PascalVOC',
    ],
    evaluation_tasks=
    'PascalVOC_Task4-1,PascalVOC_Task4-2,PascalVOC_Task4-3,PascalVOC_Task4-4',
    metrics=[
        dict(
            ann_file=
            'data/CDIOD/PascalVOC/valid/annotations_without_background.json',
            classwise=True,
            metric='bbox',
            metric_items=[
                'mAP',
                'mAP_50',
                'mAP_75',
                'mAP_s',
                'mAP_m',
                'mAP_l',
                'AR@100',
                'AR@300',
            ],
            outfile_prefix=
            '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/branches/A_original/validation_predictions',
            type='CDIODMetric',
            use_mp_eval=False),
    ],
    type='CDIODMultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
vis_cfg = dict(save_path='', type='none')
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/data/dq/cdiod-dgs/work_dirs/dtg_diagnostics/voc_ab_20261024_seed1498185232/branches/A_original'
