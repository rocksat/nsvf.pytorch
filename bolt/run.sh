DATA=${DATA}"_NSVF_format"
RES="480x640"
ARCH="nsvf_base"
SUFFIX="v1"
DATASET=/task_runtime/datasets
SAVE=$BOLT_ARTIFACT_DIR/saved/${DATA}
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_last.pt

function makedir() {
  mkdir -p $SAVE/$MODEL
  mkdir -p ${DATASET}
}

function download() {
  BLOBBY="aws --endpoint-url https://blob.mr3.simcloud.apple.com"
  S3_SRC="s3://chbucket/nsvf"
  ${BLOBBY} s3 cp ${S3_SRC}/${DATA}.tar.gz .
  tar -zxf ${DATA}.tar.gz -C ${DATASET}
  DATASET=${DATASET}/${DATA}
}

function train() {
cd nsvf.pytorch && python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..90" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 1 \
    --pixel-per-view 1024 \
    --no-preload \
    --sampling-on-mask 1.0 \
    --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "90..97" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --raymarching-stepsize-ratio 0.125 \
    --use-octree \
    --discrete-regularization \
    --color-weight 128.0 \
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
}

function render() {
cd nsvf.pytorch && python3 render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --model-overrides '{"chunk_size":256,"raymarching_tolerance":0.01}' \
    --render-beam 1 \
    --render-num-frames 90 \
    --render-save-fps 24 \
    --render-resolution $RES \
    --render-camera-poses ${DATASET}/platform_poses \
    --render-views "0..96" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "normal" \
    --render-combine-output \
    --log-format "simple"
}

# make folder
makedir

# download datasets
download

# start training on bolt
train

# start render on bolt
# render