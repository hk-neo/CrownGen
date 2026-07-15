"""SAP Encode2Points fine-tune on 우리 811명 GT 크라운.
pytorch3d-free Trainer (원본 training.py는 chamfer_distance 사용)."""
import os, sys, json, time, argparse, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'crowngen', 'external'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')

import numpy as np, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from mesh_recon.src.model import Encode2Points
from mesh_recon.src.utils import load_model_manual, load_config
from mesh_recon.src.data.tooth_dataset import ToothDataset

DPSR_CKPT = 'runs2/dpsr_weights/ours_noise_005.pt'
DPSR_CFG = 'crowngen/external/mesh_recon/configs/learning_based/noise_small/tooth_1024.yaml'
DPSR_DEFAULT = 'crowngen/external/mesh_recon/configs/default.yaml'

W_PSR, W_REG, W_NORM = 1.0, 10.0, 5.0


def load_model(dev):
    cfg = load_config(DPSR_CFG, DPSR_DEFAULT)
    model = Encode2Points(cfg).to(dev)
    ck = torch.load(DPSR_CKPT, map_location=dev, weights_only=False)
    load_model_manual(ck['state_dict'], model)
    return model, cfg


def chamfer_l2(a, b):
    """(B,N,3), (B,M,3) -> scalar mean squared distance (both dirs)."""
    dist_ab = torch.cdist(a, b, p=2)  # (B,N,M)
    nn_a = dist_ab.min(dim=2).values.mean()  # mean of nearest dist in a->b
    nn_b = dist_ab.min(dim=1).values.mean()
    return nn_a + nn_b


class TrainerLite:
    """pytorch3d-free minimal Trainer: PSR MSE + chamfer reg + normal L1."""
    def __init__(self, model, dpsr, optim, dev, cfg):
        self.model, self.dpsr, self.optim, self.dev, self.cfg = model, dpsr, optim, dev, cfg

    def step(self, batch):
        self.optim.zero_grad()
        inputs = batch['inputs'].to(self.dev)      # (B,1,1024,3) [0,1]
        gt_psr = batch['gt_psr'].to(self.dev)      # (B,1,64,64,64)
        # NEW: NaN guard for cache corrupt items
        if torch.isnan(gt_psr).any():
            return {'psr': float('nan'), 'reg': float('nan'), 'total': float('nan')}
        pred_pc, pred_n = self.model(inputs.squeeze(1))   # (B,1024,3), (B,1024,3)
        if torch.isnan(pred_pc).any() or torch.isnan(pred_n).any():
            # skip this batch — let optimizer continue without this step
            return {'psr': float('nan'), 'reg': float('nan'), 'total': float('nan')}
        # Encode2Points.forward takes (B, N, 3) point cloud
        pred_pc = torch.clamp(pred_pc, 0, 0.99)
        n_norm = pred_n.norm(-1, keepdim=True)
        # Guard against near-zero normals (model outputs magnitude ~1e-5 -> divide gives huge vals)
        # Instead: clamp the divisor so tiny normals stay small-but-bounded after div
        pred_n = pred_n / n_norm.clamp(min=1e-3)
        # PSR
        psr_grid = self.dpsr(pred_pc, pred_n)
        if self.cfg['model']['psr_tanh']:
            psr_grid = torch.tanh(psr_grid); gt_psr_t = torch.tanh(gt_psr.squeeze(1))
        else:
            gt_psr_t = gt_psr.squeeze(1)
        loss_psr = F.mse_loss(psr_grid, gt_psr_t)
        # Chamfer reg (input pc -> refined pc)
        loss_reg = chamfer_l2(inputs.squeeze(1), pred_pc)
        # normal L1 reg (refined normal -> mesh normal from cache; we have none currently)
        loss_n = pred_n.abs().mean() * 0.0  # placeholder; wire from cache norm later
        loss = W_PSR * loss_psr + W_REG * loss_reg + W_NORM * loss_n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'psr': loss_psr.item(), 'reg': loss_reg.item(), 'total': loss.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args()
    dev = torch.device('cuda')
    model, cfg = load_model(dev)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev, weights_only=False))
        print(f'resumed from {args.resume}', flush=True)
    from mesh_recon.src.dpsr import DPSR
    dpsr = DPSR(res=(64, 64, 64), sig=2).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_ds = ToothDataset('train')
    val_ds = ToothDataset('val')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))

    trainer = TrainerLite(model, dpsr, optim, dev, cfg)
    best = float('inf')
    os.makedirs('runs2', exist_ok=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        s = {'psr': 0., 'reg': 0., 'total': 0., 'n': 0}
        for batch in train_loader:
            m = trainer.step(batch)
            if not math.isnan(m['total']):
                for k in ('psr','reg','total'): s[k] += m[k]
                s['n'] += 1
        sched.step()
        # val
        model.eval(); v = 0.; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(dev); gt_psr = batch['gt_psr'].to(dev)
                pred_pc, pred_n = model(inputs.squeeze(1))
                pred_pc = torch.clamp(pred_pc, 0, 0.99)
                pred_n = pred_n / (pred_n.norm(-1, keepdim=True) + 1e-8)
                psr_grid = torch.tanh(dpsr(pred_pc, pred_n)) if cfg['model']['psr_tanh'] else dpsr(pred_pc, pred_n)
                v += F.mse_loss(psr_grid, torch.tanh(gt_psr.squeeze(1)) if cfg['model']['psr_tanh'] else gt_psr.squeeze(1)).item(); vn += 1
        v /= max(vn,1)
        print(f'ep{ep:02d} t={(time.time()-t0):.1f}s train={s["total"]/max(s["n"],1):.4f} val_psr={v:.4f}', flush=True)
        if ep % 10 == 0 or ep == args.epochs:
            torch.save(model.state_dict(), f'runs2/sap_finetuned_e{ep}.pt')
        if v < best:
            best = v
            torch.save(model.state_dict(), 'runs2/sap_finetuned_best.pt')
            print(f'  ★ best val={best:.4f}', flush=True)
    print(f'DONE best_val={best:.4f}', flush=True)

if __name__ == '__main__':
    main()
