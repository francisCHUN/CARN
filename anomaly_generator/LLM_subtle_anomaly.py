from torch_geometric.data import Data
from .anomaly_list import ANOMALY_TYPE_LIST
from .LLM_contextual_anomaly import llm_anomaly_generation_pipeline, SYSTEM_PROMPT


_DATASET_DESC = {
    "cora": "the Cora dataset (Machine Learning research paper titles/abstracts)",
    "citeseer": "the CiteSeer dataset (Computer Science research paper titles/abstracts)",
    "pubmed": "the PubMed dataset (biomedical paper titles/abstracts)",
    "arxiv": "the ogbn-arxiv dataset (arXiv Computer Science paper titles/abstracts)",
    "wikics": "the Wiki-CS dataset (Wikipedia Computer Science articles)",
}

USER_PROMPT_SUBTLE = """You are given a normal node from {dataset_desc}. \
Its original node label/topic is ({label_name}). \
Its graph neighbours mostly belong to other topics; one neighbour-suggested but \
adjacent topic that this node is NOT actually about is ({designated_label}). \
We want a SUBTLE contextual anomaly: text that is plausibly about ({designated_label}) \
while staying stylistically embedded in the original writing.

Original text attribute:
{raw_text}

Task: Rewrite the text so it concerns ({designated_label}) instead of ({label_name}), \
while REMAINING lexically and stylistically as close to the original as possible, so it \
evades simple similarity / contextual-anomaly detectors.

Requirements:
1) Topic shift: the rewritten text should clearly concern ({designated_label}), not ({label_name}).
2) Subtlety: change only what is needed to shift the topic. Keep the wording, length, \
tone and sentence structure close to the original; minimise lexical divergence. \
Do NOT rewrite from scratch.
3) Plausibility: the result must read as a natural, in-distribution node. No typos, no \
hallucinated facts, no meta-commentary, no generation markers.
4) Structure preservation: if the original is a title only, output a title only; if it is \
a title plus an abstract, output a title plus an abstract in the same proportions.
5) Output format: return only the rewritten text, formatted exactly like the original. \
Do not add labels such as "Title:" / "Abstract:" or phrases like "Rewritten text:".
"""


def _pick_desc(dataset_name: str) -> str:
    name = dataset_name.lower()
    for key, desc in _DATASET_DESC.items():
        if key in name:
            return desc
    raise ValueError(f"Dataset name: {dataset_name} is not implemented for subtle anomaly")


def llm_subtle_anomaly_generator(data: Data, dataset_name: str, n: int, anomaly_type: int, random_seed: int, k_neighbors: int) -> Data:
    if ANOMALY_TYPE_LIST[anomaly_type] != "LLM Subtle Contextual Anomaly":
        raise ValueError(f"Invalid anomaly type: {anomaly_type}")
    user_prompt = USER_PROMPT_SUBTLE.replace("{dataset_desc}", _pick_desc(dataset_name))
    return llm_anomaly_generation_pipeline(data, n, anomaly_type, random_seed, k_neighbors, user_prompt, SYSTEM_PROMPT)
