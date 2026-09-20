import random
import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST


def _gen_structural_cliques_on_normal_nodes(data: Data, m: int, n: int, random_seed: int):
    N = data.x.shape[0]
    if hasattr(data, "anomaly_labels"):
        normal_pool = (data.anomaly_labels == 0).nonzero(as_tuple=True)[0]
    else:
        normal_pool = torch.arange(N, dtype=torch.long)
    needed = m * n
    if normal_pool.numel() < needed:
        raise ValueError(f"Not enough normal nodes to host {needed} structural anomalies "
                         f"(only {normal_pool.numel()} normal nodes available).")

    gen = torch.Generator(device="cpu").manual_seed(int(random_seed))
    perm = normal_pool[torch.randperm(normal_pool.numel(), generator=gen)]
    chosen = perm[:needed]

    y_outlier = torch.zeros(N, dtype=torch.long)
    y_outlier[chosen] = 1

    new_edges = []
    for i in range(n):
        members = chosen[m * i: m * (i + 1)]
        pairs = torch.combinations(members)
        new_edges.append(pairs)
        new_edges.append(pairs.flip(1))
    new_edges = torch.cat(new_edges, dim=0).T
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return data, y_outlier


def structural_anomaly_generator(data: Data, dataset_name: str, m: int, n: int, anomaly_type: int, random_seed: int) -> Data:
    if "cora" not in dataset_name.lower() and \
        "citeseer" not in dataset_name.lower() and \
        "pubmed" not in dataset_name.lower() and \
        "wikics" not in dataset_name.lower() and "arxiv" not in dataset_name.lower():
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")
    if ANOMALY_TYPE_LIST[anomaly_type] != "Structural Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    if not hasattr(data, "edge_index"):
        raise ValueError("data must have the .edge_index attribute")

    data, y_outlier = _gen_structural_cliques_on_normal_nodes(data, m, n, random_seed)
    y_list = y_outlier.tolist()
    assert len([y for y in y_list if y == 1]) == n*m, f"The number of anomaly nodes is not equal to n*m"

    N = len(y_list)
    if hasattr(data, "anomaly_labels"):
        prev_labels = data.anomaly_labels.to(y_outlier.device).long()
    else:
        prev_labels = torch.zeros(N, dtype=torch.int64, device=y_outlier.device)
    base_types = data.anomaly_types if hasattr(data, "anomaly_types") else [0] * N
    prev_types = list(base_types) if not isinstance(base_types, list) else list(base_types)
    if len(prev_types) < N:
        prev_types = prev_types + [0] * (N - len(prev_types))

    merged_labels = (prev_labels.bool() | y_outlier.bool()).long()
    merged_types = prev_types.copy()
    added = 0
    for idx in range(N):
        if y_list[idx] == 1 and prev_labels[idx].item() != 1:
            merged_types[idx] = anomaly_type
            added += 1
    data.anomaly_labels = merged_labels
    data.anomaly_types = merged_types

    print(f"Structural anomaly generation completed: {added} new structural anomalies added "
          f"({n} cliques x {m} nodes); total anomalies now {int(merged_labels.sum().item())}")
    return data