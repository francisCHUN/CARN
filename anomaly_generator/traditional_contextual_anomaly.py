import torch
from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
from .utils import encode_text, TextEncoder
import random


def traditional_contextual_anomaly_generator(data: Data, dataset_name: str, n: int, anomaly_type: int, k: int, random_seed: int) -> Data:
    if ANOMALY_TYPE_LIST[anomaly_type] != "Traditional Contextual Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    
    if "cora" in dataset_name.lower():
        return traditional_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "citeseer" in dataset_name.lower():
        return traditional_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "pubmed" in dataset_name.lower():
        return traditional_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "arxiv" in dataset_name.lower():
        return traditional_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    elif "wikics" in dataset_name.lower():
        return traditional_anomaly_generation_pipeline(data, n, anomaly_type, k, random_seed)
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")

def traditional_anomaly_generation_pipeline(data: Data, n: int, anomaly_type: int, k: int, random_seed: int) -> Data:
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
    data = traditional_text_only_anomaly(data, selected_idxs, anomaly_type, k, random_seed)
    data = encode_text(data)
    return data

def traditional_text_only_anomaly(data: Data, selected_idxs: torch.Tensor, anomaly_type: int, k: int, random_seed: int) -> Data:
    print("Generating traditional contextual anomaly...")
    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    count = 0
    text_encoder = TextEncoder()
    for idx in selected_idxs.tolist():
        seed = random_seed + idx
        gen = torch.Generator(device=selected_idxs.device).manual_seed(int(seed))
        num_node = len(data.raw_texts)
        selected_k_idxs = torch.randperm(num_node, generator=gen, device=data.x.device)[:k]
        most_distant_idx = get_most_distant_node(data, idx, selected_k_idxs, text_encoder)
        processed_text[idx] = replace_words(data.raw_texts[idx], data.raw_texts[most_distant_idx], random_seed)
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
    print("Traditional contextual anomaly generation completed")
    return data


def get_most_distant_node(data: Data, idx: int, selected_k_idxs: torch.Tensor, text_encoder: TextEncoder) -> int:
    selected_k_idxs = selected_k_idxs.tolist()
    selected_k_text_list = [data.raw_texts[i] for i in selected_k_idxs]
    current_node_text_list = [data.raw_texts[idx]] * len(selected_k_text_list)
    current_node_embeddings = text_encoder.encode_list(current_node_text_list)
    selected_k_embeddings = text_encoder.encode_list(selected_k_text_list)
    similarity = torch.cosine_similarity(current_node_embeddings, selected_k_embeddings)
    pos = int(similarity.argmin().item())
    most_distant_idx = selected_k_idxs[pos]
    return most_distant_idx


def replace_words(text1: str, text2: str, random_seed: int) -> str:
    rng = random.Random(random_seed)
    word_list1 = text1.split()
    len_word_list1 = len(word_list1)
    word_list2 = text2.split()
    len_word_list2 = len(word_list2)
    
    if len_word_list1 == 0 or len_word_list2 == 0:
        raise ValueError("Text is empty")
    
    min_replace = max(1, len_word_list1 // 4)
    max_replace = max(min_replace, len_word_list1 // 2)
    num_words_to_replace = rng.randint(min_replace, max_replace)
    
    if num_words_to_replace > len_word_list2:
        num_words_to_replace = len_word_list2
    
    if num_words_to_replace > len_word_list1:
        num_words_to_replace = len_word_list1
    
    selected_words = rng.sample(word_list2, num_words_to_replace)
    
    max_start_idx = len_word_list1 - num_words_to_replace
    start_idx = rng.randint(0, max(0, max_start_idx))
    
    for i in range(num_words_to_replace):
        word_list1[start_idx + i] = selected_words[i]
    
    result_text = " ".join(word_list1)
    return result_text