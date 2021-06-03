# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="UDF_snapshot_76853_vid_50010020000000545_NSVF_format"
RES="480x640"
ARCH="nsvf_image"
SUFFIX="v1"
DATASET=/home/chen/Workspace/RadianceField/NSVF/nsvf.pytorch/datasets/Europa/${DATA}
SAVE=/home/chen/Workspace/RadianceField/NSVF/nsvf.pytorch/saved/$DATA
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

# start training locally
python train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..90" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 1 \
    --pixel-per-view 1024 \
    --no-preload \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "90..97" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --raymarching-stepsize-ratio 0.125 \
    --use-octree \
    --discrete-regularization \
    --color-weight 128.0 \
    --alpha-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 150000 \
    --lr 0.001 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 50 --max-update 100000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at  "5000,25000" \
    --reduce-step-size-at "5000,25000" \
    --pruning-every-steps 1000 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}
