"""Pure-torch 구현 of CrownGen/PVCNN custom CUDA ops.

공식 cg_boundary_prediction_module/modules/functional 의 CUDA 확장을 컴파일 없이
실행하기 위한 drop-in 대체. API/시그니처는 공식 코드에 일치시켰다. 속도는 CUDA
버전보다 느리지만 정확도는 동등하다.
"""
import torch
import torch.nn.functional as F

__all__ = [
    'gather', 'furthest_point_sample', 'grouping', 'ball_query',
    'avg_voxelize', 'trilinear_devoxelize', 'nearest_neighbor_interpolate',
    'kl_loss', 'huber_loss',
]


# ───────────────── sampling ─────────────────

def gather(features, indices):
    """features (B,C,N), indices (B,M) int → (B,C,M)."""
    B, C, _ = features.shape
    idx = indices.long().unsqueeze(1).expand(B, C, -1)
    return torch.gather(features, 2, idx)


def furthest_point_sample(coords, num_samples):
    """coords (B,3,N) → sampled center coords (B,3,M)."""
    B, _, N = coords.shape
    device = coords.device
    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch = torch.arange(B, device=device)
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = coords[batch, :, farthest]              # (B,3)
        d = ((coords - centroid.unsqueeze(-1)) ** 2).sum(1)  # (B,N)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return gather(coords, centroids)


# ───────────────── grouping ─────────────────

def grouping(features, indices):
    """features (B,C,N), indices (B,M,U) int → (B,C,M,U)."""
    B, C, _ = features.shape
    M, U = indices.shape[1], indices.shape[2]
    idx = indices.long().reshape(B, M * U).unsqueeze(1).expand(B, C, M * U)
    out = torch.gather(features, 2, idx)
    return out.view(B, C, M, U)


def ball_query(centers_coords, points_coords, radius, num_neighbors):
    """centers (B,3,M), points (B,3,N) → neighbor indices (B,M,U).

    반경 이내 이웃을 거리순으로 최대 num_neighbors 개. 부족하면 가장 가까운 이웃으로
    패딩(PVCNN ball_query 동작 근사).
    """
    B, _, M = centers_coords.shape
    N = points_coords.shape[2]
    d = torch.cdist(centers_coords.transpose(1, 2), points_coords.transpose(1, 2))  # (B,M,N)
    d_sorted, idx_sorted = torch.sort(d, dim=-1)                                     # (B,M,N)
    idx = idx_sorted[:, :, :num_neighbors].contiguous()                              # (B,M,U)
    # 반경 밖 이웃은 첫(가까운) 이웃으로 패딩
    valid = d_sorted[:, :, :num_neighbors] <= radius                                 # (B,M,U)
    first = idx[:, :, 0:1].expand(-1, -1, num_neighbors)
    return torch.where(valid, idx, first)


# ───────────────── voxelization ─────────────────

def avg_voxelize(features, coords, resolution):
    """features (B,C,N), coords (B,3,N) int in [0,r) → (B,C,r,r,r). scatter-mean."""
    B, C, N = features.shape
    r = resolution
    lin = coords[:, 0].long() * r * r + coords[:, 1].long() * r + coords[:, 2].long()  # (B,N)
    out = features.new_zeros(B, C, r ** 3)
    cnt = features.new_zeros(B, r ** 3)
    ones = features.new_ones(B, N)
    cnt.scatter_add_(1, lin, ones)
    lin_e = lin.unsqueeze(1).expand(B, C, N)
    out.scatter_add_(2, lin_e, features)
    out = out / cnt.unsqueeze(1).clamp(min=1.0)
    return out.view(B, C, r, r, r)


def trilinear_devoxelize(features, coords, resolution, is_training=True, tooth_mask=None):
    """features (B,C,r,r,r), coords (B,3,N) float in [0,r] → (B,C,N). 삼선형 보간.

    tooth_mask (B,) bool 이 주어지면 missing(tooth_mask=0) 치아의 출력을 0으로.
    (공식 TrilinearDevoxelization 의 tooth_mask 시맨틱)
    """
    B, C = features.shape[:2]
    N = coords.shape[2]
    r = resolution
    flat = features.view(B, C, r ** 3)
    c = coords.clone().clamp(0, r - 1.0001)
    x0 = c[:, 0].long(); y0 = c[:, 1].long(); z0 = c[:, 2].long()
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
    dx = (c[:, 0] - x0.float())[:, None]  # (B,1,N)
    dy = (c[:, 1] - y0.float())[:, None]
    dz = (c[:, 2] - z0.float())[:, None]

    def g(ix, iy, iz):
        lin = (ix * r * r + iy * r + iz)                  # (B,N)
        return torch.gather(flat, 2, lin[:, None, :].expand(B, C, N))

    c000 = g(x0, y0, z0); c100 = g(x1, y0, z0); c010 = g(x0, y1, z0); c110 = g(x1, y1, z0)
    c001 = g(x0, y0, z1); c101 = g(x1, y0, z1); c011 = g(x0, y1, z1); c111 = g(x1, y1, z1)
    out = (c000 * (1 - dx) * (1 - dy) * (1 - dz) + c100 * dx * (1 - dy) * (1 - dz) +
           c010 * (1 - dx) * dy * (1 - dz) + c110 * dx * dy * (1 - dz) +
           c001 * (1 - dx) * (1 - dy) * dz + c101 * dx * (1 - dy) * dz +
           c011 * (1 - dx) * dy * dz + c111 * dx * dy * dz)
    if tooth_mask is not None:
        out = out * tooth_mask.to(out.dtype).view(B, 1, 1)
    return out


# ───────────────── interpolation ─────────────────

def nearest_neighbor_interpolate(points_coords, centers_coords, centers_features):
    """3-최근접 이웃 역거리 가중 보간.
    points_coords (B,3,N), centers_coords (B,3,M), centers_features (B,C,M) → (B,C,N).
    """
    B, C, M = centers_features.shape
    N = points_coords.shape[2]
    d = torch.cdist(points_coords.transpose(1, 2), centers_coords.transpose(1, 2))  # (B,N,M)
    w, idx = d.topk(3, dim=-1, largest=False)                                       # (B,N,3)
    w = 1.0 / (w + 1e-8)
    w = w / w.sum(-1, keepdim=True)
    idx = idx.long().unsqueeze(1).expand(B, C, N, 3)                                # (B,C,N,3)
    gathered = torch.gather(centers_features.unsqueeze(2).expand(B, C, N, M), 3, idx)  # (B,C,N,3)
    return (gathered * w.unsqueeze(1)).sum(-1)                                      # (B,C,N)


# ───────────────── losses ─────────────────

def kl_loss(x, y):
    x = F.softmax(x.detach(), dim=1)
    y = F.log_softmax(y, dim=1)
    return torch.mean(torch.sum(x * (torch.log(x) - y), dim=1))


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.full_like(abs_error, fill_value=delta))
    losses = 0.5 * (quadratic ** 2) + delta * (abs_error - quadratic)
    return torch.mean(losses)
