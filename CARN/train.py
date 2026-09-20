
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .utils import set_seed, anneal_tau, REPO_ROOT, eval_roc_auc, eval_average_precision, \
    eval_f1, eval_precision_at_k, eval_recall_at_k, _HAVE_PYGOD
from .data_io import prepare
from .cs_eval import rank01_safe, decompose_scores
from .model import CARN
from .losses import context_recon_loss, contrastive_loss, causal_discrepancy, \
    causal_discrepancy_margin_loss, modularity_loss, collapse_loss, \
    attribute_structure_consistency_loss
from .synthetic import make_synthetic_anomalies, make_synthetic_anomalies_vec, \
    make_synthetic_anomalies_cosine, make_synthetic_anomalies_random


@torch.no_grad()
def evaluate(model, batch, opts, tau: float):
    model.eval()
    H = batch["H"]
    out = model(H, batch["edge_index"], batch["nbr_idx"], batch["nbr_mask"], tau)
    h_pred, h_pred_norm = out[0], out[1]
    h_cf = out[2]
    S_struct = out[4]
    S_attr = out[5]

    score_recon = 1.0 - F.cosine_similarity(H, h_pred_norm, dim=-1)

    s_log = (S_struct + 1e-8).log()
    a_log = (S_attr + 1e-8).log()
    score_kl = (S_struct * (s_log - a_log)).sum(dim=-1)

    no_cf = getattr(opts, "ablate", "none") in ("no_cf", "no_intervention")
    score_cf = (None if (h_cf is None or no_cf) else causal_discrepancy(h_pred, h_cf))

    score = rank01_safe(score_recon) + opts.alpha_score * rank01_safe(score_kl)
    if score_cf is not None:
        score = score + opts.beta_score * rank01_safe(score_cf)

    struct_ranked = batch.get("struct_ranked")
    if struct_ranked is not None and float(getattr(opts, "gamma_score", 0.0)) > 0.0:
        score = score + float(opts.gamma_score) * struct_ranked.to(score.device)

    label = batch["label"]
    q = float(label.sum().item()) / label.numel()
    thr = torch.quantile(score.float(), 1.0 - q)
    pred = (score >= thr).long()
    label = label.cpu()
    score = score.cpu()
    pred = pred.cpu()
    score_recon = score_recon.cpu()
    score_kl_cpu = score_kl.cpu()
    score_cf_cpu = score_cf.cpu() if score_cf is not None else None

    res = {
        "auc_score": float(eval_roc_auc(label, score.float())),
        "avg_precision": float(eval_average_precision(label, score.float())),
        "f1_score": float(eval_f1(label, pred)),
        "precision_at_k": float(eval_precision_at_k(label, score.float(), opts.k)),
        "recall_at_k": float(eval_recall_at_k(label, score.float(), opts.k)),
    }
    res["score_mean_normal"] = float(score[label == 0].mean().item())
    res["score_mean_anomaly"] = float(score[label == 1].mean().item())
    res["score_recon_normal"] = float(score_recon[label == 0].mean().item())
    res["score_recon_anomaly"] = float(score_recon[label == 1].mean().item())
    res["score_kl_normal"] = float(score_kl_cpu[label == 0].mean().item())
    res["score_kl_anomaly"] = float(score_kl_cpu[label == 1].mean().item())
    res["score_kl_mean"] = float(score_kl_cpu.mean().item())
    if score_cf_cpu is not None:
        res["score_cf_normal"] = float(score_cf_cpu[label == 0].mean().item())
        res["score_cf_anomaly"] = float(score_cf_cpu[label == 1].mean().item())
        res["score_cf_mean"] = float(score_cf_cpu.mean().item())
    else:
        res["score_cf_normal"] = None
        res["score_cf_anomaly"] = None
        res["score_cf_mean"] = None
    res["contamination"] = q
    N = S_struct.size(0); K = S_struct.size(1)
    res["collapse_ratio"] = float(torch.norm(S_struct.sum(dim=0), p='fro').item() / (N / (K ** 0.5)))
    res["n_active_clusters"] = int((S_struct.sum(dim=0) > 1e-3).sum().item())
    res["tau"] = float(tau)
    return res, score, pred


def train(opts):
    set_seed(opts.seed)
    batch = prepare(opts, version="v4")
    device = batch["device"]
    H = batch["H"]
    in_dim = H.size(1)
    K = batch["K"]

    ablate = opts.ablate
    dual_view = (ablate != "no_dual_view")
    use_local = (ablate != "no_local_context")
    use_global = (ablate != "no_global_context")

    model = CARN(in_dim=in_dim, hid_dim=opts.hid_dim, max_neigh=opts.max_neigh, K=K,
                    heads=opts.nhead, nhead=opts.nhead, num_layers=opts.num_layers,
                    gnn_layers=opts.gnn_layers, dropout=opts.dropout,
                    cf_mode=opts.cf_mode, ablate=ablate,
                    dual_view=dual_view, use_local=use_local, use_global=use_global).to(device)
    model.comm_learner.kmeans_init(H, batch["edge_index"], K, seed=opts.seed)
    if ablate == "fixed_community":
        for p in model.comm_learner.parameters():
            p.requires_grad_(False)
        print("[ablate=fixed_community] comm_learner frozen after k-means init")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=opts.lr, weight_decay=opts.weight_decay)

    gen = torch.Generator(device=device).manual_seed(opts.seed)

    batch["struct_ranked"] = None
    if float(getattr(opts, "gamma_score", 0.0)) > 0.0:
        from .cs_eval import compute_struct_ranked
        struct_ranked, struct_name = compute_struct_ranked(
            opts.struct_score, batch, opts.cc_cap)
        batch["struct_ranked"] = struct_ranked.to(device)
        print(f"[struct] gamma_score={opts.gamma_score} signal={struct_name} "
              f"cc_cap={opts.cc_cap} -> structural term active in combined score")

    big_graph = batch["N"] > 10000
    select_by = getattr(opts, "select_by", "loss")
    best_loss, best_loss_state, best_loss_epoch = float("inf"), None, -1
    best_auc, best_auc_state = -1.0, None
    frac = float(getattr(opts, "loss_window", 1.0))
    frac = min(1.0, max(0.0, frac))
    window_start = max(1, int((1.0 - frac) * opts.epochs) + 1)

    for epoch in range(1, opts.epochs + 1):
        model.train()
        opt.zero_grad()
        tau = anneal_tau(epoch, opts)
        h_pred, h_pred_norm, h_cf, h_cf_norm, S_struct, S_attr, comm_mean, Z = model(
            H, batch["edge_index"], batch["nbr_idx"], batch["nbr_mask"], tau)

        l_ctx = context_recon_loss(H, h_pred)
        no_cf = opts.ablate in ("no_cf", "no_intervention")
        no_synth = (opts.ablate == "no_synthetic_negatives")
        no_align = (opts.ablate == "no_alignment")
        random_donor = (opts.ablate == "random_donor")
        l_ctx_cf = context_recon_loss(H, h_cf) if not no_cf else None

        if no_synth:
            neg_idx, h_prime = None, None
        else:
            with torch.no_grad():
                if random_donor:
                    neg_idx, h_prime = make_synthetic_anomalies_random(H, opts.p_syn, gen)
                elif opts.donor_strategy == "cosine":
                    neg_idx, h_prime = make_synthetic_anomalies_cosine(H, S_struct, opts.p_syn, gen)
                else:
                    comm_id_dyn = S_struct.argmax(dim=1)
                    if big_graph:
                        neg_idx, h_prime = make_synthetic_anomalies_vec(H, comm_id_dyn, opts.p_syn, gen)
                    else:
                        neg_idx, h_prime = make_synthetic_anomalies(H, comm_id_dyn, opts.p_syn, gen)
        l_con = contrastive_loss(H, h_pred_norm, neg_idx, h_prime, opts.tau) \
            if not no_synth else None

        ramp = min(1.0, max(0.0, (epoch - opts.cf_warmup) / max(1, opts.cf_warmup)))
        if no_cf or no_synth:
            l_cf = None
        else:
            score_cf = causal_discrepancy(h_pred, h_cf)
            l_cf = causal_discrepancy_margin_loss(score_cf, neg_idx, opts.cf_margin)

        l_mod = modularity_loss(S_struct, batch["edge_index"], batch["deg"], batch["m"])
        l_col = collapse_loss(S_struct)
        l_asc = attribute_structure_consistency_loss(S_struct, S_attr) if not no_align else None

        loss = l_ctx
        if not no_cf:
            loss = loss + opts.lambda_ctx_cf * l_ctx_cf
        if not no_synth:
            loss = loss + opts.lambda_con * l_con
        loss = loss + opts.lambda_mod * l_mod + opts.lambda_clp * l_col
        if not no_align:
            loss = loss + opts.lambda_asc * l_asc
        if not no_cf and not no_synth:
            loss = loss + opts.lambda_cf * ramp * l_cf
        loss.backward()
        opt.step()

        cur_loss = l_ctx.item()
        if not no_cf:
            cur_loss += opts.lambda_ctx_cf * l_ctx_cf.item()
        if not no_synth:
            cur_loss += opts.lambda_con * l_con.item()
        cur_loss += opts.lambda_mod * l_mod.item() + opts.lambda_clp * l_col.item()
        if not no_align:
            cur_loss += opts.lambda_asc * l_asc.item()
        if not no_cf and not no_synth:
            cur_loss += opts.lambda_cf * 1.0 * l_cf.item()
        if epoch >= window_start and cur_loss < best_loss:
            best_loss = cur_loss
            best_loss_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss_epoch = epoch

        if epoch % 10 == 0 or epoch == 1 or epoch == opts.epochs:
            res, score, pred = evaluate(model, batch, opts, tau=tau)
            def _fmt(v, name):
                return f"{name}={v.item():.4f} " if v is not None else f"{name}=n/a "
            cf_log = (f"ctxcf={l_ctx_cf.item():.4f} cf={l_cf.item():.4f} "
                      if (not no_cf and l_cf is not None) else "ctxcf=n/a cf=n/a ")
            cf_score_log = (
                f"CF_norm={res['score_cf_normal']:.4f} CF_anom={res['score_cf_anomaly']:.4f}"
                if res.get("score_cf_normal") is not None else "CF_norm=n/a CF_anom=n/a")
            print(f"epoch {epoch:3d} tau={tau:.3f} | loss={loss.item():.4f} "
                  f"(ctx={l_ctx.item():.4f} {cf_log}"
                  f"{_fmt(l_con, 'con')}"
                  f"mod={l_mod.item():.4f} col={l_col.item():.4f} "
                  f"{_fmt(l_asc, 'asc').strip()}) | "
                  f"clps={res['collapse_ratio']:.2f} "
                  f"active={res['n_active_clusters']}/{K} | "
                  f"AUC={res['auc_score']:.4f} AUPR={res['avg_precision']:.4f} "
                  f"F1={res['f1_score']:.4f} P@{opts.k}={res['precision_at_k']:.4f} "
                  f"R@{opts.k}={res['recall_at_k']:.4f} "
                  f"| s_norm={res['score_mean_normal']:.4f} s_anom={res['score_mean_anomaly']:.4f} "
                  f"KL_norm={res['score_kl_normal']:.4f} KL_anom={res['score_kl_anomaly']:.4f} "
                  f"{cf_score_log}")
            if select_by == "auc" and res["auc_score"] > best_auc:
                best_auc = res["auc_score"]
                best_auc_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    chosen_state = best_loss_state if select_by == "loss" else best_auc_state
    if chosen_state is not None:
        model.load_state_dict(chosen_state)
    final_res, score, pred = evaluate(model, batch, opts, tau=opts.tau_comm_end)
    print(final_res)
    if select_by == "loss":
        print(f"[select] criterion=loss(min) best_loss_full={best_loss:.4f} @epoch{best_loss_epoch} "
              f"(window=epochs[{window_start}..{opts.epochs}], loss_window={frac})")
    else:
        print(f"[select] criterion=auc best_auc={best_auc:.4f}")

    s_recon_dump, s_kl_dump, s_cf_dump = decompose_scores(model, batch, opts.tau_comm_end)
    score_ablate_tag = f"_ablate_{opts.ablate}" if opts.ablate != "none" else ""
    score_path = os.path.join(
        REPO_ROOT, "experiments",
        f"score_carn_v4_{opts.dataset_name}{score_ablate_tag}_seed{opts.seed}.pt")
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    final_score = score.detach().float().cpu() if torch.is_tensor(score) \
        else torch.as_tensor(score, dtype=torch.float32)
    torch.save({
        "score": final_score,
        "score_recon": s_recon_dump,
        "score_kl": s_kl_dump,
        "score_cf": s_cf_dump,
        "label": batch["label"].detach().cpu(),
        "anomaly_types": batch["anomaly_types"].detach().cpu(),
        "dataset": opts.dataset_name, "seed": opts.seed, "ablate": opts.ablate,
        "alpha_score": opts.alpha_score, "beta_score": opts.beta_score,
        "gamma_score": float(getattr(opts, "gamma_score", 0.0)),
        "overall_auc": final_res["auc_score"], "overall_ap": final_res["avg_precision"],
    }, score_path)
    print(f"Saved final scores -> {score_path}  (N={batch['N']})")

    if getattr(opts, "save_emb", False):
        model.eval()
        with torch.no_grad():
            out = model(H, batch["edge_index"], batch["nbr_idx"], batch["nbr_mask"],
                        tau=opts.tau_comm_end)
            h_pred_norm = out[1].detach().cpu()
            Z = out[7].detach().cpu()
        emb_path = opts.emb_path or os.path.join(
            REPO_ROOT, "experiments",
            f"emb_carn_v4_{opts.dataset_name}{('_ablate_'+opts.ablate) if opts.ablate != 'none' else ''}_seed{opts.seed}.pt")
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        torch.save({
            "h_pred_norm": h_pred_norm,
            "Z": Z,
            "label": batch["label"].detach().cpu(),
            "score": score.detach().cpu() if torch.is_tensor(score) else torch.as_tensor(score),
            "dataset": opts.dataset_name,
            "seed": opts.seed,
            "hid_dim": opts.hid_dim,
        }, emb_path)
        print(f"Saved embeddings -> {emb_path}  "
              f"(N={h_pred_norm.shape[0]}, dim={h_pred_norm.shape[1]})")

    if getattr(opts, "cs_split", False):
        from .cs_eval import per_type_metrics, fmt_cs_table, structural_sweep, \
            fmt_sweep_table, serialize_sweep
        at = batch.get("anomaly_types")
        if at is None or int(at.ne(0).sum().item()) == 0:
            print("cs_split: dataset has no non-zero anomaly_types; skipping CS breakdown.")
        else:
            cs = per_type_metrics(score, batch["label"], at)
            print("\n" + "=" * 70)
            print(f"CARN per-type breakdown on {opts.dataset_name} (type-vs-normal protocol)")
            print("=" * 70)
            print(fmt_cs_table(cs))

            sweep_block = None
            gammas = [float(g) for g in opts.gamma_sweep.split()] if opts.gamma_sweep.strip() else []
            if gammas:
                sweep_rows, struct_name = structural_sweep(
                    model, batch, opts, gammas, kind=opts.struct_score, cc_cap=opts.cc_cap)
                print("\n" + "=" * 70)
                print(f"Structural-term validation sweep on {opts.dataset_name} "
                      f"(struct={struct_name}, sweep={opts.gamma_sweep})")
                print("=" * 70)
                print(fmt_sweep_table(sweep_rows, struct_name))
                sweep_block = serialize_sweep(sweep_rows, struct_name)

            cs_suffix = f"_seed{opts.seed}" if getattr(opts, "_multi_seed", False) else ""
            cs_ablate_tag = f"_ablate_{opts.ablate}" if opts.ablate != "none" else ""
            cs_out = {**{k: cs[k] for k in ("overall", "contextual", "structural", "per_type")},
                      "dataset": opts.dataset_name, "seed": opts.seed,
                      "ablate": opts.ablate,
                      "n_anomalies": int(batch['label'].sum().item())}
            if sweep_block is not None:
                cs_out["structural_sweep"] = sweep_block
            cs_path = os.path.join(REPO_ROOT,
                                   f"result_carn_v4_cs_{opts.dataset_name}{cs_ablate_tag}{cs_suffix}.json")
            with open(cs_path, "w") as f:
                json.dump(cs_out, f, indent=2)
            print(f"\nSaved CS breakdown -> {cs_path}")

    print("\n" + "=" * 70)
    if select_by == "loss":
        print(f"CARN results on {opts.dataset_name} (min-loss model @epoch{best_loss_epoch} "
              f"in window[epochs {window_start}..{opts.epochs}], full-ramp loss={best_loss:.4f})")
    else:
        print(f"CARN results on {opts.dataset_name} (best-AUC model, auc={best_auc:.4f})")
    print("=" * 70)
    print(f"  AUROC      : {final_res['auc_score']:.4f}")
    print(f"  AUPR       : {final_res['avg_precision']:.4f}")
    print(f"  F1         : {final_res['f1_score']:.4f}")
    print(f"  Precision@{opts.k}: {final_res['precision_at_k']:.4f}")
    print(f"  Recall@{opts.k}   : {final_res['recall_at_k']:.4f}")
    print(f"  mean score (normal) ={final_res['score_mean_normal']:.4f}")
    print(f"  mean score (anomaly)={final_res['score_mean_anomaly']:.4f}  "
          f"(should exceed normal)")
    print(f"  KL score (normal)   ={final_res['score_kl_normal']:.4f}")
    print(f"  KL score (anomaly)  ={final_res['score_kl_anomaly']:.4f}  "
          f"(should exceed normal)")
    if final_res.get("score_cf_normal") is not None:
        print(f"  CF score (normal)   ={final_res['score_cf_normal']:.4f}")
        print(f"  CF score (anomaly)  ={final_res['score_cf_anomaly']:.4f}  "
              f"(should exceed normal)")
    else:
        print("  CF score             : n/a (ablate=no_cf)")
    print(f"  collapse ratio={final_res['collapse_ratio']:.2f} (1.0=balanced, "
          f"{K**0.5:.1f}=total collapse)  active_clusters={final_res['n_active_clusters']}/{K}")
    print(f"  metrics via: {'pygod.metric' if _HAVE_PYGOD else 'sklearn fallback'}")

    out = {**{k: final_res[k] for k in
              ['auc_score', 'avg_precision', 'f1_score', 'precision_at_k', 'recall_at_k',
               'score_mean_normal', 'score_mean_anomaly', 'contamination',
               'score_recon_normal', 'score_recon_anomaly',
               'score_kl_normal', 'score_kl_anomaly', 'score_kl_mean',
               'score_cf_normal', 'score_cf_anomaly', 'score_cf_mean']},
           "dataset": opts.dataset_name, "epochs": opts.epochs, "hid_dim": opts.hid_dim,
           "K": K, "n_comm_arg": opts.n_comm,
           "select_by": select_by,
           "best_loss": best_loss if select_by == "loss" else None,
           "best_loss_epoch": best_loss_epoch if select_by == "loss" else None,
           "best_auc": best_auc if select_by == "auc" else None,
           "lambda_con": opts.lambda_con, "tau": opts.tau,
           "lambda_mod": opts.lambda_mod, "lambda_clp": opts.lambda_clp,
           "lambda_asc": opts.lambda_asc, "alpha_score": opts.alpha_score,
           "beta_score": opts.beta_score, "gamma_score": getattr(opts, "gamma_score", 0.0),
           "lambda_cf": opts.lambda_cf,
           "lambda_ctx_cf": opts.lambda_ctx_cf, "cf_margin": opts.cf_margin,
           "cf_warmup": opts.cf_warmup, "cf_mode": opts.cf_mode,
           "ablate": opts.ablate,
           "tau_comm_start": opts.tau_comm_start, "tau_comm_end": opts.tau_comm_end,
           "donor_strategy": opts.donor_strategy,
           "collapse_ratio": final_res["collapse_ratio"],
           "n_active_clusters": final_res["n_active_clusters"],
           "seed": opts.seed,
           "n_anomalies": int(batch['label'].sum().item())}
    suffix = f"_seed{opts.seed}" if getattr(opts, "_multi_seed", False) else ""
    ablate_tag = f"_ablate_{opts.ablate}" if opts.ablate != "none" else ""
    out_path = os.path.join(REPO_ROOT, "results", f"result_carn_v4_{opts.dataset_name}{ablate_tag}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results -> {out_path}")
    return out


def _build_parser():
    p = argparse.ArgumentParser(description="CARN: Counterfactual Context Reasoning")
    for _a in _V4_ARGS:
        p.add_argument(*_a[0], **_a[1])
    return p


def parse_args(argv=None):
    return _build_parser().parse_args(argv)


def run_args(opts):
    seeds = opts.seeds if opts.seeds else [opts.seed]
    opts._multi_seed = len(seeds) > 1
    all_out = []
    for i, s in enumerate(seeds):
        opts.seed = s
        if opts._multi_seed:
            print("\n" + "#" * 70)
            print(f"# SEED {s}  ({i + 1}/{len(seeds)})")
            print("#" * 70)
        all_out.append(train(opts))
    if opts._multi_seed:
        aggregate_results(all_out, opts)
    return all_out


def aggregate_results(all_out, opts):
    keys = ['auc_score', 'avg_precision', 'f1_score', 'precision_at_k', 'recall_at_k']
    print("\n" + "=" * 70)
    print(f"CARN multi-seed summary on {opts.dataset_name} "
          f"({len(all_out)} seeds: {[o['seed'] for o in all_out]})")
    print("=" * 70)
    summary = {"dataset": opts.dataset_name, "seeds": [o["seed"] for o in all_out],
               "per_seed": all_out}
    for k in keys:
        vals = np.array([o[k] for o in all_out], dtype=float)
        mu, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd
        print(f"  {k:15s}: {mu:.4f} +/- {sd:.4f}   (per-seed: "
              + " ".join(f"{v:.4f}" for v in vals) + ")")
    out_path = os.path.join(REPO_ROOT, "results", f"result_carn_v4_{opts.dataset_name}_multiseed.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved multi-seed summary -> {out_path}")
    return summary


_V4_ARGS = [
    (["--dataset_name"], dict(type=str, default="pubmed_fixed_sbert_mix_2_5")),
    (["--data_dir"], dict(type=str, default="data/generated")),
    (["--epochs"], dict(type=int, default=100)),
    (["--lr"], dict(type=float, default=1e-3)),
    (["--weight_decay"], dict(type=float, default=1e-5)),
    (["--hid_dim"], dict(type=int, default=128)),
    (["--nhead"], dict(type=int, default=4)),
    (["--num_layers"], dict(type=int, default=2)),
    (["--max_neigh"], dict(type=int, default=16)),
    (["--p_syn"], dict(type=float, default=0.4)),
    (["--lambda_con"], dict(type=float, default=1.0)),
    (["--tau"], dict(type=float, default=0.2)),
    (["--dropout"], dict(type=float, default=0.1)),
    (["--k"], dict(type=int, default=20)),
    (["--seed"], dict(type=int, default=42)),
    (["--seeds"], dict(type=int, nargs="*", default=None,
                       help="multi-seed run, e.g. --seeds 42 43 44; overrides --seed")),
    (["--select_by"], dict(type=str, default="loss", choices=["loss", "auc"],
                        help="model selection criterion for the final/test model: "
                             "'loss' = epoch with min global training loss (default); "
                             "'auc' = epoch with best AUC (legacy)")),
    (["--loss_window"], dict(type=float, default=0.5,
                        help="fraction of FINAL epochs used as the min-loss selection window "
                             "(0.5 = last 50%%; 1.0 = all epochs). Avoids the degenerate early "
                             "epochs where community-formation losses are still low")),
    (["--ablate"], dict(type=str, default="none",
                        choices=["none", "no_cf", "no_intervention",
                                 "no_dual_view", "fixed_community",
                                 "no_local_context", "no_global_context",
                                 "no_alignment", "no_synthetic_negatives", "random_donor"],
                        help="ablation switch (one per run): "
                             "no_cf/no_intervention=drop CF path; "
                             "no_dual_view=S_attr==S_struct; "
                             "fixed_community=freeze comm_learner after k-means init; "
                             "no_local_context=mask local(GAT) token; "
                             "no_global_context=mask community token; "
                             "no_alignment=drop l_asc; "
                             "no_synthetic_negatives=drop l_con + donor sampling; "
                             "random_donor=random donor instead of cross-community. "
                             "'none' = full model (default)")),
    (["--gpu"], dict(type=int, default=0)),
    (["--n_comm"], dict(type=int, default=-1,
                        help="number of communities K; -1 -> sqrt(N) heuristic (no Louvain)")),
    (["--gnn_layers"], dict(type=int, default=2, help="GraphEncoder GCN depth")),
    (["--tau_comm_start"], dict(type=float, default=1.0,
                        help="starting softmax temperature (k-means init already breaks symmetry)")),
    (["--tau_comm_end"], dict(type=float, default=0.2,
                        help="final (sharpest) temperature; keep >=0.1 to avoid gradient vanishing")),
    (["--anneal_epochs"], dict(type=int, default=-1, help="linear anneal horizon; -1 -> opts.epochs")),
    (["--lambda_mod"], dict(type=float, default=1.0, help="modularity loss weight")),
    (["--lambda_clp"], dict(type=float, default=0.3,
                        help="collapse loss weight (higher -> stronger anti-collapse)")),
    (["--donor_strategy"], dict(type=str, default="argmax", choices=["argmax", "cosine"],
                        help="synthetic-anomaly donor sampling")),
    (["--lambda_asc"], dict(type=float, default=0.5,
                        help="attribute-structure consistency loss weight")),
    (["--alpha_score"], dict(type=float, default=0.3,
                        help="KL score weight in combined anomaly scoring")),
    (["--beta_score"], dict(type=float, default=0.5,
                        help="causal-discrepancy score weight in combined anomaly scoring "
                             "(0 -> degenerates to v3 scoring)")),
    (["--gamma_score"], dict(type=float, default=0.5,
                        help="structural-anomaly term weight (option ①): "
                             "score += gamma_score * rank(struct_signal); 0 disables it. "
                             "struct_signal from --struct_score (clustering by default)")),
    (["--lambda_cf"], dict(type=float, default=0.5, help="counterfactual margin loss weight")),
    (["--lambda_ctx_cf"], dict(type=float, default=0.3,
                        help="counterfactual-path reconstruction loss weight")),
    (["--cf_margin"], dict(type=float, default=0.5,
                        help="margin for synthetic anomalies in causal-discrepancy loss")),
    (["--cf_warmup"], dict(type=int, default=20,
                        help="margin loss ramps in linearly over [cf_warmup, 2*cf_warmup] epochs")),
    (["--cf_mode"], dict(type=str, default="intervene", choices=["intervene", "generator"],
                        help="intervene: shared Reasoner with do(community) token swap; "
                             "generator: learned conditional generator (ablation)")),
    (["--save_emb"], dict(action="store_true",
                        help="dump node embeddings (h_pred_norm + Z) + labels to .pt for t-SNE viz")),
    (["--emb_path"], dict(type=str, default=None,
                        help="explicit output path for embeddings; default "
                             "experiments/emb_carn_v4_{dataset}_seed{seed}.pt")),
    (["--cs_split"], dict(action="store_true",
                        help="also report per-anomaly-type AUC/AP (Contextual vs Structural) "
                             "and save a CS breakdown json; see CARN/cs_eval.py")),
    (["--gamma_sweep"], dict(type=str, default="0.5 1.0 2.0 4.0",
                        help="space-separated gamma values for the structural-term validation sweep "
                             "(option ①); empty string disables the sweep")),
    (["--struct_score"], dict(type=str, default="clustering", choices=["clustering", "cc", "degree"],
                        help="structural signal to add: clustering (local CC, best clique fingerprint) "
                             "or degree")),
    (["--cc_cap"], dict(type=int, default=64,
                        help="cap on neighbor count when computing clustering coefficient "
                             "(bounds cost on hub-heavy graphs like arxiv)")),
]


if __name__ == "__main__":
    run_args(parse_args())