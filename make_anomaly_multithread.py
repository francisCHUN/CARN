import time

from anomaly_generator import traditional_contextual_anomaly_generator, global_anomaly_generator, dummy_anomaly_generator, structural_anomaly_generator
from anomaly_generator.LLM_contextual_anomaly import generate_LLM_text, TOKENS, SYSTEM_PROMPT, USER_PROMPT_CORA, USER_PROMPT_CITESEER, USER_PROMPT_PUBMED, USER_PROMPT_ARXIV, USER_PROMPT_WIKICS
from anomaly_generator.LLM_subtle_anomaly import USER_PROMPT_SUBTLE, _pick_desc
import anomaly_generator.LLM_contextual_anomaly as llm_ctx
import sohoyo_query
from anomaly_generator.utils import encode_text
from anomaly_generator.anomaly_list import ANOMALY_TYPE_LIST
from data.raw_data_loader import LLMGNNDataLoader
from datasets import load_dataset
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os
import argparse
from typing import List

K_NEIGHBORS = 3
K = 50
RANDOM_SEED = 42
M = 10
PUBMED_EACH_NUM = 1000

def save_data(data: Data, data_dir: str, dataset_name: str, anomaly_type: int, anomaly_num: int, output_name: str = None) -> None:
    if output_name:
        file_path = os.path.join(data_dir, f"{output_name}.pt")
    else:
        file_path = os.path.join(data_dir, f"{dataset_name}_{anomaly_type}_{anomaly_num}.pt")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torch.save(data, file_path)

def make_outliner_texts_pubmed():
    result = []
    outliner_breast_cancer = load_from_huggingface("Gaborandi/breast_cancer_pubmed_abstracts")
    result.extend(outliner_breast_cancer)
    outliner_HIV = load_from_huggingface("Gaborandi/HIV_pubmed_abstracts")
    result.extend(outliner_HIV)
    outliner_brain_tumor = load_from_huggingface("Gaborandi/Brain_Tumor_pubmed_abstracts")
    result.extend(outliner_brain_tumor)
    outliner_Alzheimer = load_from_huggingface("Gaborandi/Alzheimer_pubmed_abstracts")
    result.extend(outliner_Alzheimer)
    return result

def make_outliner_texts_wikics(loader: LLMGNNDataLoader):
    data_outliner = loader.load_dataset("citeseer_fixed_sbert", is_map_label=False)
    return data_outliner.raw_texts

def make_outliner_texts_cora(loader: LLMGNNDataLoader):
    data_outliner = loader.load_dataset("wikics_fixed_sbert", is_map_label=False)
    return data_outliner.raw_texts

def make_outliner_texts_citeseer(loader: LLMGNNDataLoader):
    data_outliner = loader.load_dataset("wikics_fixed_sbert", is_map_label=False)
    return data_outliner.raw_texts

def load_from_huggingface(dataset_name: str) -> List[str]:
    dataset = load_dataset(dataset_name)["train"]
    title = dataset["title"]
    abstract = dataset["abstract"]
    str_list = []
    count = 0
    for title, abstract in zip(title, abstract):
        if title is None or abstract is None:
            continue
        str_list.append("Title: "+ title + "\nAbstract: " + abstract)
        count += 1
        if count >= PUBMED_EACH_NUM:
            break
    print(f"{dataset_name} example:\n{str_list[0]}")
    return str_list


def setup_llm_backend(backend: str, model_name: str = None) -> None:
    if backend == "sohoyo":
        llm_ctx.LLM_MODEL = model_name or sohoyo_query.MODEL

        def _sohoyo_call_llm(prompt: str, model: str, token: str,
                             system_prompt: str = SYSTEM_PROMPT,
                             max_tries: int = sohoyo_query.MAX_RETRIES) -> str:
            return sohoyo_query.call_llm(prompt=prompt, model=model,
                                         system_prompt=system_prompt, max_tries=max_tries)

        llm_ctx.call_llm = _sohoyo_call_llm
        print(f"[LLM] backend=sohoyo model={llm_ctx.LLM_MODEL} (antchat token pool unused)")
    elif backend == "antchat":
        if model_name:
            llm_ctx.LLM_MODEL = model_name
        print(f"[LLM] backend=antchat model={llm_ctx.LLM_MODEL} tokens={len(TOKENS)}")
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def llm_generated_contextual_anomaly_mt(data: Data, selected_idxs: torch.Tensor, anomaly_type: int,
                                        k_neighbors: int, user_prompt: str, system_prompt: str,
                                        random_seed: int, max_workers: int) -> Data:
    print("Generating LLM-Generated Contextual anomaly...")

    processed_text = data.raw_texts.copy() if not hasattr(data, "processed_text") else data.processed_text.copy()
    anomaly_labels = torch.zeros(len(processed_text), dtype=torch.int64, device=data.x.device) if not hasattr(data, "anomaly_labels") else data.anomaly_labels.clone()
    anomaly_types = [0] * len(processed_text) if not hasattr(data, "anomaly_types") else data.anomaly_types.copy()

    idxs = selected_idxs.tolist()

    def _gen(task_index: int, idx: int):
        token = TOKENS[task_index % len(TOKENS)]
        print(f"Generating LLM-Generated Contextual anomaly for node {idx}...")
        llm_text = generate_LLM_text(data, idx, k_neighbors, user_prompt, system_prompt, random_seed, token)
        time.sleep(1)
        return idx, llm_text

    count = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_gen, i, idx): idx for i, idx in enumerate(idxs)}
        for fut in as_completed(futures):
            try:
                idx, llm_text = fut.result()
            except Exception as e:
                failed += 1
                print(f"[WARNING] node {futures[fut]} failed after retries and will be skipped (not marked as anomaly): {e}")
                continue
            processed_text[idx] = llm_text
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


def llm_anomaly_generation_pipeline_mt(data: Data, n: int, anomaly_type: int, random_seed: int,
                                       k_neighbors: int, user_prompt: str, system_prompt: str,
                                       max_workers: int) -> Data:
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
    data = llm_generated_contextual_anomaly_mt(data, selected_idxs, anomaly_type, k_neighbors,
                                               user_prompt, system_prompt, random_seed, max_workers)
    data = encode_text(data)
    return data


def llm_generated_contextual_anomaly_generator_mt(data: Data, dataset_name: str, n: int,
                                                  anomaly_type: int, random_seed: int,
                                                  k_neighbors: int, max_workers: int) -> Data:
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
    return llm_anomaly_generation_pipeline_mt(data, n, anomaly_type, random_seed, k_neighbors,
                                              user_prompt, SYSTEM_PROMPT, max_workers)


def llm_subtle_anomaly_generator_mt(data: Data, dataset_name: str, n: int, anomaly_type: int,
                                    random_seed: int, k_neighbors: int, max_workers: int) -> Data:
    if ANOMALY_TYPE_LIST[anomaly_type] != "LLM Subtle Contextual Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    user_prompt = USER_PROMPT_SUBTLE.replace("{dataset_desc}", _pick_desc(dataset_name))
    return llm_anomaly_generation_pipeline_mt(data, n, anomaly_type, random_seed, k_neighbors,
                                              user_prompt, SYSTEM_PROMPT, max_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cora_fixed_sbert")
    parser.add_argument("--anomaly_type", type=int, default=2)
    parser.add_argument("--anomaly_num", type=int, default=108)
    parser.add_argument("--data_dir", type=str, default="data/generated")
    parser.add_argument("--output_dir", type=str, default="data/generated")
    parser.add_argument("--is_map_label", type=bool, default=False)
    parser.add_argument("--output_name", type=str, default="cora_fixed_sbert_gpt_2",
                        help="Optional explicit output file stem (without .pt). Useful for stacked/mixed datasets, "
                             "e.g. cora_fixed_sbert_mix_2_8. If omitted, defaults to {dataset_name}_{anomaly_type}_{anomaly_num}.")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of worker threads for concurrent per-node LLM calls (anomaly_type 2 and 8 only).")
    parser.add_argument("--llm_backend", type=str, default="antchat", choices=["antchat", "sohoyo"],
                        help="LLM gateway used for per-node rewrites (anomaly_type 2 and 8 only): "
                             "'antchat' (default, pooled tokens, Qwen*/DeepSeek*/GLM* models) or "
                             "'sohoyo' (OpenAI-compatible gateway, gpt-*/gemini-* models).")
    parser.add_argument("--llm_model", type=str, default=None,
                        help="Optional model-name override for the selected backend. Defaults: "
                             "antchat -> LLM_MODEL env / Qwen3-235B-A22B-Instruct-2507; "
                             "sohoyo -> SOHOYO_MODEL env / gpt-5.6-luna.")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    anomaly_type = args.anomaly_type
    n=args.anomaly_num
    loader = LLMGNNDataLoader(data_dir=args.data_dir)
    data = loader.load_dataset(dataset_name, is_map_label=args.is_map_label)
    setup_llm_backend(args.llm_backend, args.llm_model)
    k_neighbors = K_NEIGHBORS
    k = K
    random_seed = RANDOM_SEED

    if anomaly_type == 1:
        data = dummy_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        n=n,
                                        anomaly_type=1,
                                        random_seed=random_seed)
    elif anomaly_type == 2:
        data = llm_generated_contextual_anomaly_generator_mt(data=data,
                                                        dataset_name=dataset_name,
                                                        n=n,
                                                        anomaly_type=2,
                                                        random_seed=random_seed,
                                                        k_neighbors=k_neighbors,
                                                        max_workers=args.max_workers)
    elif anomaly_type == 8:
        data = llm_subtle_anomaly_generator_mt(data=data,
                                            dataset_name=dataset_name,
                                            n=n,
                                            anomaly_type=8,
                                            random_seed=random_seed,
                                            k_neighbors=k_neighbors,
                                            max_workers=args.max_workers)
    elif anomaly_type == 3:
        data = traditional_contextual_anomaly_generator(data=data,
                                                        dataset_name=dataset_name,
                                                        n=n,
                                                        anomaly_type=3,
                                                        k=k,
                                                        random_seed=random_seed)

    elif anomaly_type == 4:
        if "pubmed" in dataset_name.lower():
            outliner_texts = make_outliner_texts_pubmed()
        elif "wikics" in dataset_name.lower():
            outliner_texts = make_outliner_texts_wikics(loader)
        elif "cora" in dataset_name.lower():
            outliner_texts = make_outliner_texts_cora(loader)
        elif "citeseer" in dataset_name.lower():
            outliner_texts = make_outliner_texts_citeseer(loader)
        else:
            raise ValueError(f"Dataset name: {dataset_name} is not implemented")
        data = global_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        n=n,
                                        anomaly_type=4,
                                        random_seed=random_seed,
                                        outliner_texts=outliner_texts)

    elif anomaly_type == 5:
        n_clique = (n // M) + (n % M > 0)
        print(f"Gemerate {n_clique*M} structural anomaly nodes")
        data = structural_anomaly_generator(data=data,
                                        dataset_name=dataset_name,
                                        m=M,
                                        n=n_clique,
                                        anomaly_type=5,
                                        random_seed=random_seed)
        n = n_clique*M

    print(data)
    if hasattr(data, "anomaly_types"):
        from collections import Counter
        breakdown = dict(Counter(data.anomaly_types))
        total = int(data.anomaly_labels.sum().item()) if hasattr(data, "anomaly_labels") else 0
        print(f"Anomaly types in output: {breakdown}  (total anomalies={total})")
    save_data(data, args.output_dir, dataset_name, anomaly_type, n, args.output_name)
    out_name = args.output_name if args.output_name else f"{dataset_name}_{anomaly_type}_{n}"
    print(f"Saved -> {os.path.join(args.output_dir, out_name)}.pt")
