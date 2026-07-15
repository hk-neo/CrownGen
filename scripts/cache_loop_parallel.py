"""3-worker parallel cache build. Each worker processes one patient at a time in a fresh subprocess."""
import os, sys, glob, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/hk.sim/miniconda3/envs/crown/bin/python'
ENV = {**os.environ, 'PYTHONPATH': f'{ROOT}:{ROOT}/crowngen/external:{ROOT}/scripts'}
LOG = f'{ROOT}/runs2/sap_cache_parallel.log'
CACHE = f'{ROOT}/runs2/sap_cache'
NORM = f'{ROOT}/Data/aligned_norm'
WORKERS = 3


def process_one(pid: str) -> tuple:
    """Returns (pid, n_saved, error_str)."""
    cmd = [PY, f'{ROOT}/scripts/build_sap_cache.py', '--pid', pid]
    try:
        r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            n_saved = sum(1 for _ in glob.glob(f'{CACHE}/{pid}_FDI*.npz'))
            return (pid, n_saved, '')
        return (pid, 0, r.stderr[-200:])
    except subprocess.TimeoutExpired:
        return (pid, 0, 'TIMEOUT')


def main():
    pids_all = sorted([os.path.basename(f).replace('.npz', '') for f in glob.glob(f'{NORM}/*.npz')])
    done = set(os.path.basename(f).split('_FDI')[0] for f in glob.glob(f'{CACHE}/*.npz'))
    todo = [p for p in pids_all if p not in done]
    print(f'patients: {len(pids_all)}, already cached: {len(done)}, todo: {len(todo)}', flush=True)
    t0 = time.time()
    with open(LOG, 'a') as logf:
        logf.write(f'[{time.strftime("%H:%M:%S")}] start, todo={len(todo)}\n'); logf.flush()
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = {exe.submit(process_one, pid): pid for pid in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, n, err = fut.result()
            line = f'[{i}/{len(todo)}] {pid} saved={n}'
            if err: line += f' ERR={err[:80]}'
            print(line, flush=True)
            with open(LOG, 'a') as logf:
                logf.write(line + '\n'); logf.flush()
            if i % 50 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(todo) - i) / rate
                print(f'  progress: {i}/{len(todo)} ({100*i/len(todo):.1f}%) rate={rate:.2f}/s ETA={remaining/60:.0f}min', flush=True)
    print(f'DONE in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()