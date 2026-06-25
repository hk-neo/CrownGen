import json, numpy as np
from pathlib import Path
from collections import defaultdict

with open("Data/SourceC_Teeth3DS/train_val_split.json") as f:
    splits = json.load(f)

for split_name in ["stage1_train", "stage1_val", "stage2_train", "stage2_val"]:
    pids = splits[split_name]
    complete = 0
    partial = 0
    n_teeth_all = []
    missing_count = 0
    
    for pid in pids:
        npz = Path(f"Data/processed/{pid}.npz")
        if not npz.exists():
            missing_count += 1
            continue
        d = np.load(npz)
        n = len(d.get("upper_labels", np.array([]))) + len(d.get("lower_labels", np.array([])))
        n_teeth_all.append(n)
        if n == 28:
            complete += 1
        else:
            partial += 1
    
    arr = np.array(n_teeth_all) if n_teeth_all else np.array([0])
    print(f"=== {split_name}: {len(pids)}명 ===")
    print(f"  전처리됨: {len(n_teeth_all)}, 누락: {missing_count}")
    print(f"  완전 치열(28): {complete}명 ({complete/max(len(n_teeth_all),1)*100:.1f}%)")
    print(f"  부분 무치아(<28): {partial}명 ({partial/max(len(n_teeth_all),1)*100:.1f}%)")
    if len(n_teeth_all):
        print(f"  치아 수: 평균 {arr.mean():.1f}, 범위 {arr.min()}~{arr.max()}")
    print()
