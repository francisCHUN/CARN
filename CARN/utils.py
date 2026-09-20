
import os
import random

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu: int) -> torch.device:
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cpu")


def anneal_tau(epoch: int, opts):
    horizon = opts.anneal_epochs if opts.anneal_epochs and opts.anneal_epochs > 0 else opts.epochs
    prog = min(1.0, max(0.0, (epoch - 1) / max(1, horizon)))
    return opts.tau_comm_start + prog * (opts.tau_comm_end - opts.tau_comm_start)


try:
    from pygod.metric import eval_roc_auc, eval_average_precision, eval_f1, \
        eval_precision_at_k, eval_recall_at_k
    _HAVE_PYGOD = True
except Exception:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    def eval_roc_auc(label, score): return float(roc_auc_score(label.cpu().numpy(), score.cpu().numpy()))
    def eval_average_precision(label, score): return float(average_precision_score(label.cpu().numpy(), score.cpu().numpy()))
    def eval_f1(label, pred): return float(f1_score(label.cpu().numpy(), pred.cpu().numpy()))
    def _pk(label, score, k, kind):
        y = label.cpu().numpy(); s = score.cpu().numpy()
        k = max(1, min(int(k), len(s)))
        order = np.argsort(-s)[:k]
        hit = y[order].sum()
        tot = min(k, y.sum())
        return float(hit / tot) if tot > 0 else 0.0
    def eval_precision_at_k(label, score, k=None): return _pk(label, score, k, "p")
    def eval_recall_at_k(label, score, k=None): return _pk(label, score, k, "r")
    _HAVE_PYGOD = False