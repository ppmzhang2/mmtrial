# MMDetection Trial

## Environment

Install MMDetection and dependencies:

```bash
# CUDA 12.4 + Python 3.11
conda env create -f conda-mmdet-py311.yaml
```

Uninstall:

```bash
conda env remove --name py311-cuda124-openmmlab
```

## Faster R-CNN

Model list:

- `faster-rcnn_r50_fpn_carafe_1x_coco`
- `faster-rcnn_r50_fpn_dconv_c3-c5_1x_coco`
- `faster-rcnn_r50_fpn_dpool_1x_coco`
- `faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco`
- `faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco`
- `faster-rcnn_r50_fpn_mdconv_c3-c5_1x_coco`
- `faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco`
- `faster-rcnn_r50_fpn_mdpool_1x_coco`
- `faster-rcnn_r50_fpn_attention_1111_1x_coco`
- `faster-rcnn_r50_fpn_attention_0010_1x_coco`
- `faster-rcnn_r50_fpn_attention_1111_dcn_1x_coco`
- `faster-rcnn_r50_fpn_attention_0010_dcn_1x_coco`
- `faster-rcnn_r50-caffe-c4_1x_coco`
- `faster-rcnn_r50-caffe-c4_mstrain_1x_coco`
- `faster-rcnn_r50-caffe-dc5_1x_coco`
- `faster-rcnn_r50-caffe_fpn_1x_coco`
- `faster-rcnn_r50_fpn_1x_coco`
- `faster-rcnn_r50_fpn_fp16_1x_coco`
- `faster-rcnn_r50_fpn_2x_coco`
- `faster-rcnn_r101-caffe_fpn_1x_coco`
- `faster-rcnn_r101_fpn_1x_coco`
- `faster-rcnn_r101_fpn_2x_coco`
- `faster-rcnn_x101-32x4d_fpn_1x_coco`
- `faster-rcnn_x101-32x4d_fpn_2x_coco`
- `faster-rcnn_x101-64x4d_fpn_1x_coco`
- `faster-rcnn_x101-64x4d_fpn_2x_coco`
- `faster-rcnn_r50_fpn_iou_1x_coco`
- `faster-rcnn_r50_fpn_giou_1x_coco`
- `faster-rcnn_r50_fpn_bounded_iou_1x_coco`
- `faster-rcnn_r50-caffe-dc5_mstrain_1x_coco`
- `faster-rcnn_r50-caffe-dc5_mstrain_3x_coco`
- `faster-rcnn_r50-caffe_fpn_ms-2x_coco`
- `faster-rcnn_r50-caffe_fpn_ms-3x_coco`
- `faster-rcnn_r50_fpn_mstrain_3x_coco`
- `faster-rcnn_r101-caffe_fpn_ms-3x_coco`
- `faster-rcnn_r101_fpn_ms-3x_coco`
- `faster-rcnn_x101-32x4d_fpn_ms-3x_coco`
- `faster-rcnn_x101-32x8d_fpn_ms-3x_coco`
- `faster-rcnn_x101-64x4d_fpn_ms-3x_coco`
- `faster-rcnn_r50_fpn_tnr-pretrain_1x_coco`
- `faster-rcnn_r50_fpg_crop640-50e_coco`
- `faster-rcnn_r50_fpg-chn128_crop640-50e_coco`
- `faster-rcnn_r50_fpn_gn_ws-all_1x_coco`
- `faster-rcnn_r101_fpn_gn-ws-all_1x_coco`
- `faster-rcnn_x50-32x4d_fpn_gn-ws-all_1x_coco`
- `faster-rcnn_x101-32x4d_fpn_gn-ws-all_1x_coco`
- `faster-rcnn_r50_fpn_groie_1x_coco`
- `faster-rcnn_hrnetv2p-w18-1x_coco`
- `faster-rcnn_hrnetv2p-w18-2x_coco`
- `faster-rcnn_hrnetv2p-w32-1x_coco`
- `faster-rcnn_hrnetv2p-w32_2x_coco`
- `faster-rcnn_hrnetv2p-w40-1x_coco`
- `faster-rcnn_hrnetv2p-w40_2x_coco`
- `faster-rcnn_r50_fpn_32x2_1x_openimages`
- `faster-rcnn_r50_fpn_32x2_1x_openimages_challenge`
- `faster-rcnn_r50_fpn_32x2_cas_1x_openimages`
- `faster-rcnn_r50_fpn_32x2_cas_1x_openimages_challenge`
- `faster-rcnn_r50_pafpn_1x_coco`
- `faster-rcnn_regnetx-3.2GF_fpn_1x_coco`
- `faster-rcnn_regnetx-3.2GF_fpn_2x_coco`
- `faster-rcnn_regnetx-400MF_fpn_ms-3x_coco`
- `faster-rcnn_regnetx-800MF_fpn_ms-3x_coco`
- `faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco`
- `faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco`
- `faster-rcnn_regnetx-4GF_fpn_ms-3x_coco`
- `faster-rcnn_res2net-101_fpn_2x_coco`
- `faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco`
- `faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco`
- `faster-rcnn_r50_fpn_rsb-pretrain_1x_coco`
- `faster-rcnn_r50_fpn_gn-all_scratch_6x_coco`

Download model with `mim`:

```bash
mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest .
```