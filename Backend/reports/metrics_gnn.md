# GNN (domain-IP graph) results

Generated 2026-10-01T20:13:27.662246+00:00 - seed 42, CPU, 17.56s, 23382 parameters.

## Limitation (read first)

> **The graph is built from URL strings only - there is NO live DNS resolution. Only URLs whose host is already an IPv4 literal produce an IP node, so the IP/subnet relations are almost empty and the graph is dominated by domain<->domain structural edges (shared subdomain token, shared TLD, token containment). These metrics therefore measure how much a GNN can do with name structure alone. Adding passive/historical DNS data (or a resolver) is the single highest-value change to this branch.**

Measured on this run: 0 of 6512 edges are infrastructure (domain-IP / subnet) and only 0 of 3343 nodes (0.00%) have any infrastructure edge at all. 6512 edges are domain<->domain structure.

## Test metrics (node level)

| metric | GCN | majority-class baseline | delta |
| --- | --- | --- | --- |
| accuracy | 0.6853 | 0.6406 | 0.0447 |
| balanced accuracy | 0.6338 | 0.5000 | 0.1338 |
| F1 (phishing) | 0.7688 | 0.7809 | -0.0121 |
| MCC | 0.2862 | 0.0000 | 0.2862 |
| recall (phishing) | 0.8169 | 1.0000 | - |
| specificity | 0.4508 | 0.0000 | - |
| ROC-AUC | 0.6783 | 0.5000 | - |
| PR-AUC | 0.7616 | 0.6406 | - |
| recall @ FPR 0.01 | 0.0203 | 0.0000 | - |
| Brier | 0.2198 | 0.3594 | - |

Test nodes: 537. The baseline predicts class 1 for every node (training nodes are 69.60% phishing), so its recall and F1 are inflated by construction - it buys phishing recall by calling everything phishing and pays for it with specificity 0.0000. Judge the model on accuracy, balanced accuracy and MCC, which a constant predictor cannot game.

## Split integrity

- node-level split derived from `data/splits/` (grouped by registered domain): 2296 train / 502 val / 537 test labelled nodes
- cross-checked against `domain_split.csv`: 0 domain(s) disagree (3327 checked)
- a domain in train cannot be a test node, because the node *is* the domain
- graph construction never reads a label (see `app/preprocessing/graph_build.py`)

## Graph

- nodes: 3343 (3327 domain, 8 IP, 8 subnet)
- edges: 6512
- edge types: {"domain_ip": 0, "domain_subnet": 0, "shared_subdomain": 2504, "shared_tld": 3162, "containment": 830, "shared_ip": 0, "ip_subnet": 16}
- URLs sampled: {"train": 2800, "val": 600, "test": 600} of {"train": 164760, "val": 35305, "test": 35305}

## Training

- epochs run 30/30 (early stopped: False), best epoch 15
- selection metric: **validation loss** (0.3767), phishing pos_weight 0.4368

## Verdict

The GCN beats the majority-class baseline on accuracy, balanced accuracy and MCC. That is a real, if modest, gain: the graph branch adds signal the constant baseline does not have. It is still far below the URL CharCNN-BiLSTM branch, and with zero infrastructure edges it is a measurement of name structure alone, not of shared hosting.
