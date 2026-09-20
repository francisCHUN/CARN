import random

import torch

from .utils import eval_roc_auc, eval_average_precision

CONTEXTUAL_TYPES = {1, 2, 3, 4, 8}
STRUCTURAL_TYPES = {5}


def _auc_ap_on_subset(score, pos_mask, normal_mask):
    keep = pos_mask | normal_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(normal_mask.sum().item())
    if keep.sum().item() == 0 or n_pos == 0 or n_neg == 0:
        return None, None, n_pos, n_neg
    s = score[keep].float()
    y = pos_mask[keep].long()
    auc = eval_roc_auc(y, s)
    ap = eval_average_precision(y, s)
    return auc, ap, n_pos, n_neg


def per_type_metrics(score, label, anomaly_types):
    score = torch.as_tensor(score).float().cpu()
    at = torch.as_tensor(anomaly_types).long().cpu()
    binary_label = torch.as_tensor(label).long().cpu()
    normal_mask = at == 0

    res = {}

    overall_pos = binary_label == 1
    auc, ap, np_, nn_ = _auc_ap_on_subset(score, overall_pos, normal_mask)
    res["overall"] = {"auc": auc, "ap": ap, "n_pos": np_, "n_neg": nn_,
                      "ratio_pct": 100 * np_ / (np_ + nn_) if (np_ + nn_) else 0.0}

    ctx_mask = torch.zeros_like(normal_mask)
    for t in CONTEXTUAL_TYPES:
        ctx_mask |= (at == t)
    auc, ap, np_, nn_ = _auc_ap_on_subset(score, ctx_mask, normal_mask)
    res["contextual"] = {"auc": auc, "ap": ap, "n_pos": np_, "n_neg": nn_,
                         "ratio_pct": 100 * np_ / (np_ + nn_) if (np_ + nn_) else 0.0}

    str_mask = torch.zeros_like(normal_mask)
    for t in STRUCTURAL_TYPES:
        str_mask |= (at == t)
    auc, ap, np_, nn_ = _auc_ap_on_subset(score, str_mask, normal_mask)
    res["structural"] = {"auc": auc, "ap": ap, "n_pos": np_, "n_neg": nn_,
                         "ratio_pct": 100 * np_ / (np_ + nn_) if (np_ + nn_) else 0.0}

    detail = {}
    for t in sorted(set(at.tolist()) - {0}):
        pm = at == t
        auc, ap, np_, nn_ = _auc_ap_on_subset(score, pm, normal_mask)
        group = "contextual" if t in CONTEXTUAL_TYPES else (
            "structural" if t in STRUCTURAL_TYPES else "other")
        detail[int(t)] = {"auc": auc, "ap": ap, "n_pos": np_, "n_neg": nn_, "group": group}
    res["per_type"] = detail
    return res


def fmt_cs_table(cs):
    rows = [
        ("Overall", cs["overall"]),
        ("Contextual", cs["contextual"]),
        ("Structural", cs["structural"]),
    ]
    lines = [f"  {'Type':<12} {'AUC':>8} {'AP':>8} {'#Pos':>6} {'#Neg':>7} {'Ratio%':>7}"]
    for name, r in rows:
        if r["auc"] is None:
            lines.append(f"  {name:<12} {'n/a':>8} {'n/a':>8} {r['n_pos']:>6} "f"{r['n_neg']:>7} {r['ratio_pct']:>6.2f}%  (skipped: pos or neg empty)")
            continue
        lines.append(f"  {name:<12} {r['auc']:>8.4f} {r['ap']:>8.4f} {r['n_pos']:>6} "f"{r['n_neg']:>7} {r['ratio_pct']:>6.2f}%")
    detail = cs.get("per_type", {})
    if detail:
        lines.append("  -- per present type --")
        for t in sorted(detail.keys()):
            d = detail[t]
            auc = "n/a" if d["auc"] is None else f"{d['auc']:.4f}"
            ap = "n/a" if d["ap"] is None else f"{d['ap']:.4f}"
            lines.append(f"   type {t} [{d['group']:<11}] AUC={auc} AP={ap}  #Pos={d['n_pos']} #Neg={d['n_neg']}")
    return "\n".join(lines)


def rank01(x):
    x = torch.as_tensor(x).float().cpu().reshape(-1)
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    order = torch.argsort(x, stable=True)
    ranks = torch.empty(n, dtype=torch.float)
    ranks[order] = torch.arange(n, dtype=torch.float)
    return ranks / (n - 1)


def rank01_safe(x):
    x = torch.as_tensor(x).float().cpu().reshape(-1)
    if x.numel() <= 1 or float(x.std().item()) < 1e-9:
        return torch.zeros_like(x)
    return rank01(x)


def local_clustering_coefficient(edge_index, num_nodes, cap=64, seed=0):
    ei = edge_index.cpu()
    src, dst = ei.tolist()
    adj = [set() for _ in range(num_nodes)]
    for a, b in zip(src, dst):
        if a != b and a < num_nodes and b < num_nodes:
            adj[b].add(int(a))
    deg = torch.tensor([len(a) for a in adj], dtype=torch.float)
    cc = torch.zeros(num_nodes, dtype=torch.float)
    rng = random.Random(seed)
    sampling_used = 0
    for v in range(num_nodes):
        ns = list(adj[v])
        k = len(ns)
        if k < 2:
            continue
        if k > cap:
            ns = rng.sample(ns, cap)
            k = cap
            sampling_used += 1
        e = 0
        for i in range(k):
            ai = adj[ns[i]]
            for j in range(i + 1, k):
                if ns[j] in ai:
                    e += 1
        cc[v] = 2.0 * e / (k * (k - 1))
    if sampling_used:
        print(f"[CC] sampled {sampling_used} high-degree nodes (cap={cap}) for clustering-CC")
    return cc, deg


@torch.no_grad()
def decompose_scores(model, batch, tau):
    import torch.nn.functional as F
    from .losses import causal_discrepancy
    model.eval()
    H = batch["H"]
    out = model(H, batch["edge_index"], batch["nbr_idx"], batch["nbr_mask"], tau)
    h_pred, h_pred_norm = out[0], out[1]
    h_cf = out[2]
    S_struct, S_attr = out[4], out[5]
    score_recon = 1.0 - F.cosine_similarity(H, h_pred_norm, dim=-1)
    s_log = (S_struct + 1e-8).log()
    a_log = (S_attr + 1e-8).log()
    score_kl = (S_struct * (s_log - a_log)).sum(dim=-1)
    score_cf = None if h_cf is None else causal_discrepancy(h_pred, h_cf)
    return score_recon.detach().cpu(), score_kl.detach().cpu(), (score_cf.detach().cpu() if score_cf is not None else None)


def _struct_signal(kind, batch, cc_cap):
    if kind in ("clustering", "cc"):
        cc, deg = local_clustering_coefficient(batch["edge_index"], batch["N"], cap=cc_cap)
        return cc, "local_clustering_coef"
    if kind == "degree":
        return batch["deg"].detach().float().cpu(), "degree"
    raise ValueError(f"unknown struct_score kind: {kind}")


def compute_struct_ranked(kind, batch, cc_cap):
    struct, name = _struct_signal(kind, batch, cc_cap)
    return rank01(struct), name


def structural_sweep(model, batch, opts, gammas, kind="clustering", cc_cap=64):
    s_recon, s_kl, s_cf = decompose_scores(model, batch, opts.tau_comm_end)
    struct, struct_name = _struct_signal(kind, batch, cc_cap)
    label = batch["label"].detach().cpu()
    at = batch["anomaly_types"].detach().cpu()

    r_recon = rank01_safe(s_recon)
    r_kl = rank01_safe(s_kl)
    r_struct = rank01_safe(struct)
    r_cf = rank01_safe(s_cf) if s_cf is not None else None
    gamma_prod = float(getattr(opts, "gamma_score", 0.0))
    base = r_recon + opts.alpha_score * r_kl + gamma_prod * r_struct
    if r_cf is not None:
        base = base + opts.beta_score * r_cf

    rows = [(f"prod(g={gamma_prod})", 0.0, per_type_metrics(base, label, at))]
    for g in gammas:
        combined = base + g * r_struct
        rows.append((f"+g_add={g}", float(g), per_type_metrics(combined, label, at)))
    return rows, struct_name


def _f(v):
    return 0.0 if v is None else float(v)


def fmt_sweep_table(rows, struct_name):
    header = (f"  {'config':<14} {'OverAUC':>8} {'OverAP':>8} "
              f"{'CtxAUC':>8} {'CtxAP':>8} {'StrAUC':>8} {'StrAP':>8}")
    lines = [header, "-" * len(header)]
    for name, g, cs in rows:
        o, c, s = cs["overall"], cs["contextual"], cs["structural"]
        lines.append(
            f"  {name:<14} {_f(o['auc']):>8.4f} {_f(o['ap']):>8.4f} "
            f"{_f(c['auc']):>8.4f} {_f(c['ap']):>8.4f} "
            f"{_f(s['auc']):>8.4f} {_f(s['ap']):>8.4f}")
    lines.append(f"  (struct signal = {struct_name}; components rank-normalized to [0,1]; "
                 f"base = production formula incl. wired-in gamma_score; rows = ADDITIONAL struct weight)")
    return "\n".join(lines)


def serialize_sweep(rows, struct_name):
    out = []
    for name, g, cs in rows:
        o, c, s = cs["overall"], cs["contextual"], cs["structural"]
        out.append({
            "config": name, "gamma": g,
            "overall_auc": o["auc"], "overall_ap": o["ap"],
            "contextual_auc": c["auc"], "contextual_ap": c["ap"],
            "structural_auc": s["auc"], "structural_ap": s["ap"],
        })
    return {"struct_signal": struct_name, "rows": out}