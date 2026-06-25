#!/bin/bash
# ① gen2k vs gen3k_ep3000 CD 비교 (GPU0) → ② pseudo-crown 재생성 (2 GPU 샤드).
# nohup 분리용 오케스트레이터. 세션 꺼져도 끝까지 감.
cd /home/hk.sim/Projects/CrownGen
LOG=runs2/orch.log
echo "[orch] START $(date '+%m-%d %H:%M')" > $LOG

# ① CD 비교
echo "[orch] ① CD 비교 시작 (GPU0)" >> $LOG
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8 python3 scripts/compare_gen_2k_3k.py --n_patients 8 > runs2/compare_2k_3k_ep3000.log 2>&1
echo "[orch] ① 완료 $(date '+%H:%M')" >> $LOG

# ② pseudo-crown 재생성 — 2 GPU 샤드
OUT=Data/processed_stage2_3k
mkdir -p "$OUT"
cp Data/processed_stage2/split.json "$OUT/split.json" 2>/dev/null
echo "[orch] ② pseudo 재생성 시작 (2 GPU 샤드)" >> $LOG
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.8 python3 scripts/gen_stage2_pseudo.py \
  --gen_ckpt runs2/gen3k_ep3000.pt --bound_ckpt runs2/boundary_official_long.pt \
  --out_dir "$OUT" --shard 0 --nshards 2 > runs2/pseudo3k_shard0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-12.8 python3 scripts/gen_stage2_pseudo.py \
  --gen_ckpt runs2/gen3k_ep3000.pt --bound_ckpt runs2/boundary_official_long.pt \
  --out_dir "$OUT" --shard 1 --nshards 2 > runs2/pseudo3k_shard1.log 2>&1 &
P1=$!
echo "[orch] ② shards PID $P0 $P1" >> $LOG
wait $P0 $P1
echo "[orch] ② 완료 $(date '+%m-%d %H:%M')" >> $LOG
N=$(ls Data/processed_stage2_3k/*.npz 2>/dev/null | wc -l)
echo "[orch] 최종 npz 수: $N" >> $LOG
echo "[orch] ALL DONE" >> $LOG
