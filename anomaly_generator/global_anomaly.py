import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
from .utils import encode_text
import random
from typing import List


def global_anomaly_generator(data: Data, dataset_name: str, n: int, outliner_texts: List[str], anomaly_type: int, random_seed: int) -> Data:
    if ANOMALY_TYPE_LIST[anomaly_type] != "Global Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    
    if "cora" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "citeseer" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "pubmed" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "arxiv" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    elif "wikics" in dataset_name.lower():
        return global_anomaly_generation_pipeline(data, n, anomaly_type, outliner_texts, random_seed)
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")

def global_anomaly_generation_pipeline(data: Data, n: int, anomaly_type: int, outliner_texts: List[str], random_seed: int) -> Data:
    raw_texts = data.raw_texts
    node_num = len(raw_texts)
    if hasattr(data, "anomaly_labels"):
        normal_idxs = (data.anomaly_labels == 0).nonzero(as_tuple=True)[0]
    else:
        normal_idxs = torch.arange(node_num)
    if normal_idxs.shape[0] < n:
        raise ValueError(f"The number of normal nodes is less than n: {normal_idxs.shape[0]} < {n}")
    gen = torch.Generator(device=normal_idxs.device).manual_seed(int(random_seed))
    perm = torch.randperm(normal_idxs.numel(), generator=gen, device=normal_idxs.device)
    selected_idxs = normal_idxs[perm[:n]]
    data = global_anomaly(data, selected_idxs, anomaly_type, outliner_texts, random_seed)
    data = encode_text(data)
    return data

def global_anomaly(data: Data, selected_idxs: torch.Tensor, anomaly_type: int, outliner_texts: List[str], random_seed: int) -> Data:
    print("Generating global anomaly...")
    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    count = 0
    for idx in selected_idxs.tolist():
        seed = random_seed + idx
        rng = random.Random(seed)
        outliner_text = rng.choice(outliner_texts)
        processed_text[idx] = outliner_text
        anomaly_labels[idx] = 1
        anomaly_types[idx] = anomaly_type
        count += 1
        if count % 100 == 0:
            print("--------------------------------")
            print(f"Generated {count} nodes")
            print("--------------------------------")

    data.processed_text = processed_text
    data.anomaly_labels = anomaly_labels
    data.anomaly_types = anomaly_types
    print("Global anomaly generation completed")
    return data
