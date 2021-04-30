DATA="UDF_snapshot_76970_vid_50010020000000774_NSVF_format"
RES="480x640"
ARCH="nsvf_base"
SUFFIX="v1"
DATASET=/task_runtime/datasets
SAVE=/task_runtime/saved/${DATA}
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL
mkdir -p ${DATASET}

# download dataset
BLOBBY="aws --endpoint-url https://blob.mr3.simcloud.apple.com"
S3_SRC="s3://chbucket/nsvf"
${BLOBBY} s3 cp ${S3_SRC}/${DATA}.tar.gz .
tar -zxf ${DATA}.tar.gz -C ${DATASET}
DATASET=${DATASET}/${DATA}

# start training on bolt
python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..90" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 2 \
    --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 1.0 \
    --sampling-patch-size 5 \
    --valid-view-resolution $RES \
    --valid-views "90..97" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
    --background-stop-gradient \
    --load-mask \
    --load-depth \
    --arch $ARCH \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --raymarching-stepsize-ratio 0.125 \
    --use-octree \
    --discrete-regularization \
    --color-weight 128.0 \
    --depth-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 150000 \
    --lr 0.001 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 2 \
    --seed 2 \
    --save-interval-updates 500 --max-update 100000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at  "5000,25000,75000" \
    --reduce-step-size-at "5000,25000,75000" \
    --pruning-every-steps 2500 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}