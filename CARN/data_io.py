
import os
import math
import random

import torch
from torch_geometric.utils import to_undirected, degree

from data.raw_data_loader import LLMGNNDataLoader
from .utils import REPO_ROOT, get_device


def load_data(data_dir: str, dataset_name: str):
    loader = LLMGNNDataLoader(data_dir=data_dir)
    data = loader.load_dataset(dataset_name, is_map_label=False)
    assert hasattr(data, "updated_x"), "dataset has no .updated_x — run make_anomaly.py (type 2) first"
    assert hasattr(data, "anomaly_labels"), "dataset has no .anomaly_labels"
    return data


def build_neighbor_index(edge_index, num_nodes: int, max_neigh: int):
    ei = to_undirected(edge_index).cpu()
    src, dst = ei.tolist()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        if s != d:
            adj[d].append(s)
    nbr_idx = torch.full((num_nodes, max_neigh), -1, dtype=torch.long)
    nbr_mask = torch.zeros(num_nodes, max_neigh, dtype=torch.bool)
    rng = random.Random(0)
    for v in range(num_nodes):
        neighs = adj[v]
        if len(neighs) > max_neigh:
            neighs = rng.sample(neighs, max_neigh)
        for j, u in enumerate(neighs):
            nbr_idx[v, j] = u
            nbr_mask[v, j] = True
    return nbr_idx, nbr_mask


def prepare(opts, version: str = "v4"):
    data = load_data(opts.data_dir, opts.dataset_name)
    N = data.x.size(0)
    device = get_device(opts.gpu)

    H = data.updated_x.float().to(device)
    x_orig = data.x.float().to(device)
    label = data.anomaly_labels.long().to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    if hasattr(data, "anomaly_types"):
        anomaly_types = torch.as_tensor(data.anomaly_types, dtype=torch.long, device=device)
    else:
        anomaly_types = torch.zeros(N, dtype=torch.long, device=device)

    cache_path = os.path.join(REPO_ROOT, "data",
                              f"carn_{version}_cache_{opts.dataset_name}_{opts.max_neigh}.pt")
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        nbr_idx = cache["nbr_idx"]
        nbr_mask = cache["nbr_mask"]
    else:
        nbr_idx, nbr_mask = build_neighbor_index(data.edge_index, N, opts.max_neigh)
        torch.save({"nbr_idx": nbr_idx, "nbr_mask": nbr_mask}, cache_path)

    nbr_idx = nbr_idx.to(device)
    nbr_mask = nbr_mask.to(device)

    if opts.n_comm and opts.n_comm > 0:
        K = int(opts.n_comm)
    else:
        K = max(2, int(round(math.sqrt(N))))

    deg = degree(edge_index[0], num_nodes=N, dtype=torch.float).to(device)
    m = edge_index.size(1) / 2.0

    n_anom = int(label.sum().item())
    print("=" * 70)
    print(f"[CARN_{version}] Dataset: {opts.dataset_name}  | nodes={N}  edges={edge_index.size(1)}")
    print(f"emb_dim={H.size(1)}  K(communities)={K}  anomalies={n_anom} "
          f"({100*n_anom/N:.2f}%)  device={device}")
    if version == "v5":
        print(f"intervention strength: alpha ~ U({opts.alpha_min}, {opts.alpha_max})  "
              f"-> severity = 1 - alpha in [{1-opts.alpha_max:.2f}, {1-opts.alpha_min:.2f}]")
    print(f"neighbor coverage: mean={nbr_mask.float().mean().item():.2f} "
          f"max={nbr_mask.sum(1).max().item()} of {opts.max_neigh}")
    print("=" * 70)

    return {
        "H": H, "x_orig": x_orig, "label": label, "edge_index": edge_index,
        "nbr_idx": nbr_idx, "nbr_mask": nbr_mask, "K": K, "deg": deg, "m": m,
        "device": device, "N": N, "anomaly_types": anomaly_types,
    }