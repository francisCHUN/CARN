import os
import uuid
import random
import time
import requests
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from .anomaly_list import ANOMALY_TYPE_LIST
from .utils import encode_text, count_label_occurrence
from typing import List
from .prompts import SYSTEM_PROMPT, USER_PROMPT_CORA, USER_PROMPT_CITESEER, USER_PROMPT_PUBMED, USER_PROMPT_ARXIV, USER_PROMPT_WIKICS
from collections import defaultdict
from typing import Dict

TOKENS = [
    'YOUR_TOKEN_HERE',
]
LLM_URL = "https://antchat.alipay.com/v1/chat/completions"
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3-235B-A22B-Instruct-2507")

MAX_TRIES = int(os.environ.get("LLM_MAX_TRIES", "5"))
RETRY_BACKOFF = 2.0

_token_counter = 0


def _next_token() -> str:
    global _token_counter
    token = TOKENS[_token_counter % len(TOKENS)]
    print(f"[LLM] Using token #{_token_counter % len(TOKENS)} for this call")
    _token_counter += 1
    return token


def call_llm(prompt: str, model: str, token: str, system_prompt: str = SYSTEM_PROMPT, max_tries: int = MAX_TRIES) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    req = {"model": model, "messages": messages, "stream": False, "enable_sec_check": False}
    last_err = None
    for attempt in range(1, max_tries + 1):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "SOFA-RpcId": "0.1",
            "SOFA-TraceId": str(uuid.uuid4()),
        }
        try:
            resp = requests.post(LLM_URL, headers=headers, json=req, timeout=120)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            last_err = e
            print(f"[LLM] request failed (attempt {attempt}/{max_tries}): {e}")
            if attempt < max_tries:
                sleep_time = min(RETRY_BACKOFF * (2 ** (attempt - 1)), 60.0)
                print(f"[LLM] retrying in {sleep_time:.1f}s ...")
                time.sleep(sleep_time)
    raise RuntimeError(f"LLM call failed after {max_tries} attempts: {last_err}") from last_err


def llm_generated_contextual_anomaly_generator(data: Data, dataset_name: str, n: int, anomaly_type: int, random_seed: int, k_neighbors: int) -> Data:
    if ANOMALY_TYPE_LIST[anomaly_type] != "LLM-Generated Contextual Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    if "cora" in dataset_name.lower():
        user_prompt = USER_PROMPT_CORA
    elif "citeseer" in dataset_name.lower():
        user_prompt = USER_PROMPT_CITESEER
    elif "pubmed" in dataset_name.lower():
        user_prompt = USER_PROMPT_PUBMED
    elif "arxiv" in dataset_name.lower():
        user_prompt = USER_PROMPT_ARXIV
    elif "wikics" in dataset_name.lower():
        user_prompt = USER_PROMPT_WIKICS
    else:
        raise ValueError(f"Dataset name: {dataset_name} is not implemented")
    return llm_anomaly_generation_pipeline(data, n, anomaly_type, random_seed, k_neighbors, user_prompt, SYSTEM_PROMPT)

def llm_anomaly_generation_pipeline(data: Data, n: int, anomaly_type: int, random_seed: int, k_neighbors: int, user_prompt: str, system_prompt: str) -> Data:
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
    data = llm_generated_contextual_anomaly(data, selected_idxs, anomaly_type, k_neighbors, user_prompt, system_prompt, random_seed)
    data = encode_text(data)
    return data

def llm_generated_contextual_anomaly(data: Data, selected_idxs: torch.Tensor, anomaly_type: int, k_neighbors: int, user_prompt: str, system_prompt: str, random_seed: int) -> Data:
    print("Generating LLM-Generated Contextual anomaly...")

    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    count = 0
    failed = 0
    for idx in selected_idxs.tolist():
        print(f"Generating LLM-Generated Contextual anomaly for node {idx}...")
        try:
            LLM_text = generate_LLM_text(data, idx, k_neighbors, user_prompt, system_prompt, random_seed, _next_token())
        except Exception as e:
            failed += 1
            print(f"[WARNING] node {idx} failed after retries and will be skipped (not marked as anomaly): {e}")
            continue
        processed_text[idx] = LLM_text
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
    print(f"LLM-Generated Contextual anomaly generation completed: {count} generated, {failed} skipped due to failures")
    return data

def generate_LLM_text(data: Data, idx: int, k_neighbors: int, user_prompt: str, system_prompt: str, random_seed: int, token: str) -> str:
    label_names = data.label_names.copy()

    node_label = data.category_names[idx]
    if node_label in label_names:
        label_names.remove(node_label)
    
    for i in range(k_neighbors):
        neighbors = get_k_hop_neighbors(data, idx, i+1)
        sorted_label_names = count_and_sort_label_names(data, neighbors)
        index = 0
        while (len(label_names) > 1 and index < len(sorted_label_names)):
            current_label_name = sorted_label_names[index]
            if current_label_name in label_names:
                label_names.remove(current_label_name)
            index += 1
        if len(label_names) == 1:
            break
        
    label_occurrence: Dict[str, int] = count_label_occurrence(data)
    label_occurrence = {label: occurrence for label, occurrence in label_occurrence.items() if label in label_names}
    rng = random.Random(random_seed+idx)
    selected_label_name = rng.choices(list(label_occurrence.keys()), weights=list(label_occurrence.values()), k=1)[0]

    node_label_name = data.category_names[idx]
    raw_texts = data.raw_texts[idx]
    user_prompt = user_prompt.format(label_name=node_label_name, designated_label=selected_label_name, raw_text=raw_texts)
    print("="*100)
    return call_llm(prompt=user_prompt, system_prompt=system_prompt, model=LLM_MODEL, token=token)

def get_k_hop_neighbors(data: Data, idx: int, k: int)-> List[int]:
    if k == 0:
        return []
    subset_k, _, _, _ = k_hop_subgraph(idx, k, data.edge_index)
    if k == 1:
        exact_k = subset_k[subset_k != idx].tolist()
        return exact_k
    subset_k_1, _, _, _ = k_hop_subgraph(idx, k-1, data.edge_index)
    set_k   = set(subset_k.tolist())
    set_k_1 = set(subset_k_1.tolist())
    exact_k = list(set_k - set_k_1 - {idx})
    return exact_k


def count_and_sort_label_names(data: Data, neighbors: List[int]) -> List[str]:
    label_name_count = defaultdict(int)
    for neighbor in neighbors:
        label_name_count[data.category_names[neighbor]] += 1
    sorted_label_name_count = sorted(label_name_count.items(), key=lambda x: x[1], reverse=True)
    return [label_name for label_name, _ in sorted_label_name_count]