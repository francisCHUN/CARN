
import torch
import torch.nn.functional as F


def context_recon_loss(H, h_hat):
    return F.mse_loss(h_hat, H)


def contrastive_loss(H, h_hat_norm, neg_idx, h_prime, tau: float):
    pos = (F.cosine_similarity(H[neg_idx], h_hat_norm[neg_idx], dim=-1)) / tau
    neg = (F.cosine_similarity(h_prime, h_hat_norm[neg_idx], dim=-1)) / tau
    return (-pos + torch.logaddexp(pos, neg)).mean()


def causal_discrepancy(h_pred, h_cf):
    return 1.0 - F.cosine_similarity(h_pred, h_cf, dim=-1)


def causal_discrepancy_margin_loss(score_cf, neg_idx, margin: float, severity_true=None):
    mask = torch.ones(score_cf.size(0), dtype=torch.bool, device=score_cf.device)
    mask[neg_idx] = False
    normal_term = score_cf[mask].mean()
    syn_hinge = F.relu(margin - score_cf[neg_idx])
    if severity_true is None:
        syn_term = syn_hinge.mean()
    else:
        syn_term = (severity_true * syn_hinge).mean()
    return normal_term + syn_term


def severity_calibration_loss(pred_sev, neg_idx, severity_true):
    mask = torch.ones(pred_sev.size(0), dtype=torch.bool, device=pred_sev.device)
    mask[neg_idx] = False
    normal_term = (pred_sev[mask] ** 2).mean()
    syn_term = ((pred_sev[neg_idx] - severity_true) ** 2).mean()
    return normal_term + syn_term


def modularity_loss(S, edge_index, deg, m, eps: float = 1e-8):
    src, dst = edge_index[0], edge_index[1]
    A_term = (S[src] * S[dst]).sum()
    dS = (S * deg.unsqueeze(1)).sum(dim=0)
    deg_term = (dS * dS).sum()
    Q = (A_term - deg_term / (2.0 * m)) / (2.0 * m + eps)
    return -Q


def collapse_loss(S):
    K = S.size(1)
    N = S.size(0)
    return (K ** 0.5 / N) * torch.norm(S.sum(dim=0), p='fro') - 1.0


def attribute_structure_consistency_loss(S_struct, S_attr, eps: float = 1e-8):
    s_log = (S_struct + eps).log()
    a_log = (S_attr + eps).log()
    kl = (S_struct * (s_log - a_log)).sum(dim=-1)
    return kl.mean()