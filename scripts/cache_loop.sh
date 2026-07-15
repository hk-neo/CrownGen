#!/bin/bash
# Loop over all patients in Data/aligned_norm/, call build_sap_cache.py per patient
# Resume-safe: build_sap_cache.py already skips existing .npz files

set -e
export PYTHONPATH=".:crowngen/external:scripts"
PY=/home/hk.sim/miniconda3/envs/crown/bin/python
NORM=Data/aligned_norm
LOG=runs2/sap_cache_loop.log

mkdir -p runs2/sap_cache
echo "[$(date +%H:%M:%S)] loop start" >> $LOG

count=0
for f in $(ls $NORM/*.npz | sort); do
  pid=$(basename $f .npz)
  if [ -n "$(ls runs2/sap_cache/${pid}_FDI*.npz 2>/dev/null | head -1)" ]; then
    continue  # already done
  fi
  echo "[$(date +%H:%M:%S)] $pid" >> $LOG
  $PY scripts/build_sap_cache.py --pid $pid >> $LOG 2>&1 || echo "  WARN $pid failed" >> $LOG
  count=$((count+1))
  if [ $((count % 20)) -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] progress: $count patients this session, total $(ls runs2/sap_cache | wc -l) / 17800" >> $LOG
  fi
done
echo "[$(date +%H:%M:%S)] loop done, $count patients" >> $LOG
