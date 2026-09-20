
import torch
import torch.nn.functional as F


def make_synthetic_anomalies_random(H, p_syn: float, generator: torch.Generator):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    donor = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    blend = 0.5
    h_prime = blend * H[neg_idx] + (1 - blend) * H[donor]
    return neg_idx, h_prime


def make_synthetic_anomalies(H, comm_id, p_syn: float, generator: torch.Generator):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    src = comm_id[neg_idx]
    donor = torch.empty(n_syn, dtype=torch.long, device=H.device)
    all_idx = torch.arange(N, device=H.device)
    for i in range(n_syn):
        mask = comm_id != src[i]
        cand = mask.nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            cand = all_idx[all_idx != neg_idx[i]]
        j = int(torch.randint(0, cand.numel(), (1,), generator=generator, device=H.device).item())
        donor[i] = cand[j]
    blend = 0.5
    h_prime = blend * H[neg_idx] + (1 - blend) * H[donor]
    return neg_idx, h_prime


def make_synthetic_anomalies_vec(H, comm_id, p_syn: float, generator: torch.Generator):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    src = comm_id[neg_idx]
    donor = torch.randint(0, N, (n_syn,), generator=generator, device=H.device)
    collide = comm_id[donor] == src
    for _ in range(4):
        if not collide.any():
            break
        donor[collide] = torch.randint(0, N, (int(collide.sum().item()),),
                                       generator=generator, device=H.device)
        collide = comm_id[donor] == src
    if collide.any():
        donor[collide] = (donor[collide] + 1) % N
    blend = 0.5
    h_prime = blend * H[neg_idx] + (1 - blend) * H[donor]
    return neg_idx, h_prime


def make_synthetic_anomalies_cosine(H, S, p_syn: float, generator: torch.Generator,
                                    n_cand: int = 32):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    S = F.normalize(S, dim=-1)
    src_s = S[neg_idx]
    cand = torch.randint(0, N, (n_syn, n_cand), generator=generator, device=H.device)
    cand_s = S[cand]
    sim = (cand_s * src_s.unsqueeze(1)).sum(dim=-1)
    donor = cand.gather(1, sim.argmin(dim=1, keepdim=True)).squeeze(1)
    blend = 0.5
    h_prime = blend * H[neg_idx] + (1 - blend) * H[donor]
    return neg_idx, h_prime


def _sample_strength(n_syn, generator, device, alpha_min, alpha_max):
    u = torch.rand(n_syn, generator=generator, device=device)
    return alpha_min + (alpha_max - alpha_min) * u


def make_synthetic_anomalies_sev(H, comm_id, p_syn: float, generator: torch.Generator,
                                 alpha_min: float = 0.3, alpha_max: float = 0.9):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    src = comm_id[neg_idx]
    donor = torch.empty(n_syn, dtype=torch.long, device=H.device)
    all_idx = torch.arange(N, device=H.device)
    for i in range(n_syn):
        mask = comm_id != src[i]
        cand = mask.nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            cand = all_idx[all_idx != neg_idx[i]]
        j = int(torch.randint(0, cand.numel(), (1,), generator=generator, device=H.device).item())
        donor[i] = cand[j]
    alpha = _sample_strength(n_syn, generator, H.device, alpha_min, alpha_max)
    h_prime = alpha.unsqueeze(1) * H[neg_idx] + (1 - alpha).unsqueeze(1) * H[donor]
    return neg_idx, h_prime, alpha


def make_synthetic_anomalies_vec_sev(H, comm_id, p_syn: float, generator: torch.Generator,
                                     alpha_min: float = 0.3, alpha_max: float = 0.9):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    src = comm_id[neg_idx]
    donor = torch.randint(0, N, (n_syn,), generator=generator, device=H.device)
    collide = comm_id[donor] == src
    for _ in range(4):
        if not collide.any():
            break
        donor[collide] = torch.randint(0, N, (int(collide.sum().item()),),
                                       generator=generator, device=H.device)
        collide = comm_id[donor] == src
    if collide.any():
        donor[collide] = (donor[collide] + 1) % N
    alpha = _sample_strength(n_syn, generator, H.device, alpha_min, alpha_max)
    h_prime = alpha.unsqueeze(1) * H[neg_idx] + (1 - alpha).unsqueeze(1) * H[donor]
    return neg_idx, h_prime, alpha


def make_synthetic_anomalies_cosine_sev(H, S, p_syn: float, generator: torch.Generator,
                                        n_cand: int = 32,
                                        alpha_min: float = 0.3, alpha_max: float = 0.9):
    N = H.size(0)
    n_syn = max(1, int(p_syn * N))
    neg_idx = torch.randperm(N, generator=generator, device=H.device)[:n_syn]
    S = F.normalize(S, dim=-1)
    src_s = S[neg_idx]
    cand = torch.randint(0, N, (n_syn, n_cand), generator=generator, device=H.device)
    cand_s = S[cand]
    sim = (cand_s * src_s.unsqueeze(1)).sum(dim=-1)
    donor = cand.gather(1, sim.argmin(dim=1, keepdim=True)).squeeze(1)
    alpha = _sample_strength(n_syn, generator, H.device, alpha_min, alpha_max)
    h_prime = alpha.unsqueeze(1) * H[neg_idx] + (1 - alpha).unsqueeze(1) * H[donor]
    return neg_idx, h_prime, alpha