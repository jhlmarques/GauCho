_base_ = './r3det_r50_fpn_kld_6x_hrsc_rr_le90.py'

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        type='RotatedATSSHead',
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128])),
    train_cfg=dict(
        s0=dict(
            assigner=dict(
                _delete_=True,
                type='ATSSObbAssigner',
                topk=9,
                angle_version=angle_version,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))