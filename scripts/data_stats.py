import numpy as np
from pathlib import Path
from crowngen.data.fdi import get_functional_group
from collections import defaultdict

data_dir = Path("Data/processed")
files = sorted(data_dir.glob("*.npz"))

group_counts = defaultdict(int)
group_boundary = defaultdict(list)

for f in files:
    d = np.load(f)
    for key in d.files:
        if key.endswith("_bound"):
            fdi = int(key.split("_")[1])
            group = get_functional_group(fdi)
            group_counts[group] += 1
            group_boundary[group].append(d[key])

print("=== 치아 유형별 통계 ===")
print(f"{'Type':<12} {'Count':>8} {'Ratio':>8}  {'Radius':>16}  {'Height':>16}")
print("-" * 66)
total = sum(group_counts.values())
for group in ["incisor", "canine", "premolar", "molar"]:
    cnt = group_counts[group]
    pct = cnt / total * 100
    bnds = np.array(group_boundary[group])
    r_mean, r_std = bnds[:, 3].mean(), bnds[:, 3].std()
    h_mean, h_std = bnds[:, 4].mean(), bnds[:, 4].std()
    print(f"{group:<12} {cnt:>8} {pct:>7.1f}%  {r_mean:.2f} +/- {r_std:.2f}  {h_mean:.2f} +/- {h_std:.2f}")
print("-" * 66)
print(f"{'Total':<12} {total:>8}  100.0%")
print(f"\nData: {len(files)} patients, {total} teeth, {total*1024:,} points")
