#!/usr/bin/env bash
set -u

PROJECT=/homex/cll2/cll3/Mask2Former-main
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_HOST="$PROJECT/output_offline/Q3_empirical_evidence/M4_frozen_test_${STAMP}"
RUN_CONTAINER="/workspace/Mask2Former-main/output_offline/Q3_empirical_evidence/M4_frozen_test_${STAMP}"
mkdir -p "$RUN_HOST"
printf '%s\n' "$RUN_HOST" > "$PROJECT/output_offline/Q3_empirical_evidence/latest_m4_frozen_test_run.txt"

LOG="$RUN_HOST/launcher.log"
exec >>"$LOG" 2>&1

echo "run=$RUN_HOST"
echo "protocol=frozen test; B3407 primary baseline; B42+B3407 mask-NMS primary candidate; consensus diagnostic"
echo "No previous cardd_test result was found during preflight."

cat > "$RUN_HOST/protocol.json" <<'JSON'
{
  "split": "cardd_test",
  "test_images": 374,
  "primary_single_model": "baseline seed3407 final checkpoint",
  "secondary_model": "baseline seed42 final checkpoint",
  "primary_candidate": "class-aware mask-NMS",
  "diagnostic_candidate": "fixed instance consensus",
  "mask_iou_threshold": 0.5,
  "max_detections_per_image": 100,
  "checkpoint_selection": "final checkpoint fixed before test",
  "post_test_tuning_allowed": false,
  "config_sha256": "7a4ca83a55f6f26e4e056b773913f5264442a5d7dc4e240636341a10362ea45d",
  "seed42_checkpoint_sha256": "4087fd55edd47c925a42bf0ec6653a0d92e16e67290e6cc867fc37379ed19214",
  "seed3407_checkpoint_sha256": "8eecd3e4f82131f03622d3c2d4425b576bc4e09b9c190a71d0571b33117f9812",
  "test_annotation_sha256": "130cde81a32820cd83091e78bdbce236753c7675306da004b88bf54bad130a71"
}
JSON

echo "Direct-start gate: physical GPU 3 used memory <= 15000 MiB."
stable=0
while [ "$stable" -lt 1 ]; do
    line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits --id=3 | head -1)
    used=$(printf '%s' "$line" | cut -d, -f1 | tr -d ' ')
    util=$(printf '%s' "$line" | cut -d, -f2 | tr -d ' ')
    temp=$(printf '%s' "$line" | cut -d, -f3 | tr -d ' ')
    if [ "$used" -le 15000 ]; then
        stable=$((stable + 1))
    else
        stable=0
    fi
    echo "$(date '+%F %T') used_mib=$used util_percent=$util temp_c=$temp stable=$stable/1"
    [ "$stable" -ge 1 ] || sleep 60
done

if docker exec m2f_gpu3 bash -lc "pgrep -af '[t]rain_cardd.py'"; then
    echo "Refusing to start: another train_cardd.py process exists in m2f_gpu3."
    printf 'PRECHECK_BUSY\n' > "$RUN_HOST/status.txt"
    exit 20
fi

docker exec m2f_gpu3 bash -lc "
  cd /workspace/Mask2Former-main &&
  python -m py_compile train_cardd.py offline_tools/compare_fusion_rules.py &&
  test -f CarDD_COCO/annotations/instances_test2017.json &&
  test -f output_offline/B00_formal_R50/20260819_133145_seed42/model_final.pth &&
  test -f output_offline/B00_formal_R50/20260822_073123_seed3407/model_final.pth
"
preflight=$?
printf '%s\n' "$preflight" > "$RUN_HOST/preflight_exit_code.txt"
if [ "$preflight" -ne 0 ]; then
    printf 'PREFLIGHT_FAILED\n' > "$RUN_HOST/status.txt"
    exit "$preflight"
fi

echo "Starting frozen seed3407 test inference."
docker exec m2f_gpu3 bash -lc "
  cd /workspace/Mask2Former-main &&
  CUDA_VISIBLE_DEVICES=0 python train_cardd.py --num-gpus 1 --eval-only \\
    --config-file configs/cardd/maskformer2_R50_cardd_baseline.yaml \\
    MODEL.WEIGHTS output_offline/B00_formal_R50/20260822_073123_seed3407/model_final.pth \\
    DATASETS.TEST \"('cardd_test',)\" \\
    OUTPUT_DIR ${RUN_CONTAINER}/seed3407 \\
    DATALOADER.NUM_WORKERS 2 \\
    > ${RUN_CONTAINER}/seed3407_console.log 2>&1
"
seed3407_status=$?
printf '%s\n' "$seed3407_status" > "$RUN_HOST/seed3407_exit_code.txt"
if [ "$seed3407_status" -ne 0 ]; then
    printf 'SEED3407_FAILED\n' > "$RUN_HOST/status.txt"
    exit "$seed3407_status"
fi

echo "Starting frozen seed42 test inference."
docker exec m2f_gpu3 bash -lc "
  cd /workspace/Mask2Former-main &&
  CUDA_VISIBLE_DEVICES=0 python train_cardd.py --num-gpus 1 --eval-only \\
    --config-file configs/cardd/maskformer2_R50_cardd_baseline.yaml \\
    MODEL.WEIGHTS output_offline/B00_formal_R50/20260819_133145_seed42/model_final.pth \\
    DATASETS.TEST \"('cardd_test',)\" \\
    OUTPUT_DIR ${RUN_CONTAINER}/seed42 \\
    DATALOADER.NUM_WORKERS 2 \\
    > ${RUN_CONTAINER}/seed42_console.log 2>&1
"
seed42_status=$?
printf '%s\n' "$seed42_status" > "$RUN_HOST/seed42_exit_code.txt"
if [ "$seed42_status" -ne 0 ]; then
    printf 'SEED42_FAILED\n' > "$RUN_HOST/status.txt"
    exit "$seed42_status"
fi

echo "Running the two predeclared output-space rules exactly once."
docker exec m2f_gpu3 bash -lc "
  cd /workspace/Mask2Former-main &&
  python offline_tools/compare_fusion_rules.py \\
    --base-results ${RUN_CONTAINER}/seed3407/inference/coco_instances_results.json \\
    --other-results ${RUN_CONTAINER}/seed42/inference/coco_instances_results.json \\
    --gt-json CarDD_COCO/annotations/instances_test2017.json \\
    --output-dir ${RUN_CONTAINER}/fusion \\
    --match-iou 0.5 \\
    --max-dets 100 \\
    --rules mask_nms,consensus \\
    > ${RUN_CONTAINER}/fusion_console.log 2>&1
"
fusion_status=$?
printf '%s\n' "$fusion_status" > "$RUN_HOST/fusion_exit_code.txt"
if [ "$fusion_status" -ne 0 ]; then
    printf 'FUSION_FAILED\n' > "$RUN_HOST/status.txt"
    exit "$fusion_status"
fi

printf 'COMPLETE\n' > "$RUN_HOST/status.txt"
printf '0\n' > "$RUN_HOST/exit_code.txt"
echo "M4 frozen test complete."
