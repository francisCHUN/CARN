# CARN

Anomaly detection on text-attributed graphs (TAGs) with counterfactual context reasoning.

CARN jointly learns:

- **Community structure** — an adaptive community learner discovers soft communities under dual structural/attribute views.
- **Context reasoning** — a context reasoner attends over local (neighbor) and global (community) context tokens.
- **Counterfactual reasoning** — node representations are contrasted against do(community) interventions to score causal discrepancy.
- **Synthetic negatives** — cross-community donor sampling provides self-supervised anomaly supervision.

## Project Structure

```
CARN/
├── CARN/                        # model, losses, training, evaluation
│   ├── model.py                 # CARN model and components
│   ├── train.py                 # training loop and CLI
│   ├── losses.py                # loss functions
│   ├── data_io.py               # data preparation utilities
│   ├── cs_eval.py               # per-anomaly-type evaluation
│   ├── synthetic.py             # synthetic anomaly donor sampling
│   └── utils.py
├── anomaly_generator/           # anomaly injection for TAGs
├── data/                        # data loading (raw_data_loader.py, label_map.py)
├── main.py                      # training entry point
├── make_anomaly_multithread.py  # anomaly dataset generation entry point
└── requirements.txt
```

## Installation

```bash
conda create -n carn python=3.11
conda activate carn
pip install -r requirements.txt
```

## Data

Place LLMGNN-format `.pt` files under `data/raw/`. The datasets follow the
[LLMGNN repository](https://github.com/CurryTang/LLMGNN/tree/master) format:

```
data/raw/cora_fixed_sbert.pt
data/raw/citeseer_fixed_sbert.pt
data/raw/pubmed_fixed_sbert.pt
data/raw/wikics_fixed_sbert.pt
data/raw/arxiv_fixed_sbert.pt
```

Node texts are encoded with a SentenceTransformer (`all-MiniLM-L6-v2` by
default). Download the model and place it at
`model/sentence-transformers/all-MiniLM-L6-v2`, or change the default
`model_name` in `anomaly_generator/utils.py`.

## Anomaly Generation

Generate benchmark datasets by injecting anomalies:

```bash
python make_anomaly_multithread.py \
    --dataset_name cora_fixed_sbert \
    --anomaly_type 5 \
    --anomaly_num 108 \
    --data_dir data/raw \
    --output_dir data/generated \
    --output_name cora_fixed_sbert_mix_2_5
```

Supported anomaly types (see `anomaly_generator/anomaly_list.py`):

| Type | Name |
|------|------|
| 1 | Dummy Anomaly |
| 2 | LLM-Generated Contextual Anomaly |
| 3 | Traditional Contextual Anomaly |
| 4 | Global Anomaly |
| 5 | Structural Anomaly |
| 8 | LLM Subtle Contextual Anomaly |

Types 2 and 8 call an external LLM API to rewrite node texts; configure the
endpoint and credentials in `anomaly_generator/LLM_contextual_anomaly.py`
(`LLM_URL`, `TOKENS`, `LLM_MODEL`). Types 1/3/4/5 run locally without LLM
access. Anomaly types can be stacked to build mixed benchmarks (set a shared
`--output_name` so later types load the file produced by earlier ones).

## Training

```bash
# single seed
python main.py --dataset_name cora_fixed_sbert_mix_2_5

# multi-seed run with mean/std aggregation
python main.py --seeds 42 43 44 --dataset_name pubmed_fixed_sbert_mix_2_5

# per-anomaly-type (contextual vs structural) breakdown
python main.py --dataset_name cora_fixed_sbert_mix_2_5 --cs_split
```

Each run reports AUC / AP / F1 / P@K / R@K and writes JSON results to
`results/`. Training data is read from `--data_dir` (default
`data/generated`), so generated anomaly datasets are ready to use as-is.

Key hyperparameters: `--hid_dim`, `--n_comm`, `--tau`, `--lambda_cf`,
`--max_neigh`, `--epochs`, `--lr`. Run `python main.py --help` for the full
list, including ablation switches:

```
--ablate no_cf | no_intervention | no_dual_view | fixed_community |
          no_local_context | no_global_context | no_alignment |
          no_synthetic_negatives | random_donor
```

## Acknowledgements

- The `anomaly_generator` module is copied from
  [TAG_AD](https://github.com/Flanders1914/TAG_AD).
- The datasets follow the
  [LLMGNN](https://github.com/CurryTang/LLMGNN/tree/master) format.
