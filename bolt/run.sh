RES="480x640"
ARCH="nsvf_image_bg"
SUFFIX="v1"
DATASET=/mnt/task_runtime/datasets
SAVE=$BOLT_ARTIFACT_DIR/saved/
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_last.pt

function makedir() {
  mkdir -p $SAVE/$MODEL
  mkdir -p ${DATASET}
}

function download_europa() {
  BLOBBY="aws --endpoint-url https://blob.mr3.simcloud.apple.com"
  S3_SRC="s3://chbucket/nsvf"

  for DATA in `cat europa.txt`
  do
    ${BLOBBY} s3 cp ${S3_SRC}/${DATA}.tar.gz .
    tar -zxf ${DATA}.tar.gz -C ${DATASET}
    rm ${DATA}.tar.gz
  done
  OBJ_ID_FILE="/mnt/task_runtime/europa.txt"
}

function download_syn10() {
  BLOBBY="aws --endpoint-url https://blob.mr3.simcloud.apple.com"
  ${BLOBBY} s3 cp s3://yxw/data/image-nsvf/syn10/syn10.tar.gz .
  tar -zxf syn10.tar.gz -C ${DATASET}
  rm syn10.tar.gz
  DATASET=${DATASET}/syn10/data
  OBJ_ID_FILE=${DATASET}/train_ids.txt
}

function train() {
cd nsvf.pytorch && CUDA_VISIBLE_DEVICES=0 python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --object-id-path ${OBJ_ID_FILE} \
    --min-color 0 \
    --train-views "0..90" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 1 \
    --pixel-per-view 1024 \
    --no-preload \
    --load-mask \
    --sampling-on-mask 1.0 \
    --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "90..97" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --voxel-size 0.1 \
    --raymarching-stepsize 0.002 \
    --discrete-regularization \
    --color-weight 10.0 \
    --alpha-weight 1.0 \
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
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL} \
    | tee $SAVE/train.log
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
download_europa

# start training on bolt
train

# start render on bolt
# render
