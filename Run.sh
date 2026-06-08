docker exec perceive bash -lc '
cd /workspace && python3 src/utils/scripts/run_freespace_pipeline.py \
  --input-video /workspace/data/demo/eg.mp4 \
  --output-dir /workspace/output/freespace_pipeline_alpha \
  --fusion-mode alpha \
  --alpha 0.7 \
  --non-free-threshold 0.2 \
  --history-size 3 \
  --history-decay 0.6 \
  --device cuda \
  --yolop-repo /workspace/src/YOLOP/external/YOLOP \
  --yolop-checkpoint /workspace/weights/YOLOP/End-to-end.pth \
  --yolop-img-size 640 \
  --mask-mode probability \
  --sea-raft-repo /workspace/src/SEA_RAFT/external/SEA-RAFT \
  --sea-raft-config /workspace/src/SEA_RAFT/external/SEA-RAFT/config/eval/spring-S.json \
  --sea-raft-url MemorySlices/Tartan-C-T-TSKH-spring540x960-S \
  --save-frames \
  --save-arrays \
  --vis-threshold 0.5 \
  --vis-binary \
  --overwrite
'
