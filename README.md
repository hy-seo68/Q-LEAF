# Q-LEAF: Query-centric Label-aware Entry Adaptation for Filtered Approximate Nearest Neighbor Search

<!-- ![VLDB](https://img.shields.io/badge/VLDB-Artifacts_Available-blue) -->

Q-LEAF is a framework for optimizing filtered approximate nearest neighbor search (Filtered ANNS) through label-specific subgraph construction and query-adaptive entry point selection. By pre-computing topology-aware entry points for each label partition and dynamically selecting the optimal entry point at query time, Q-LEAF significantly reduces search latency while maintaining high recall.

---

## Overview

Q-LEAF operates in a three-stage pipeline:

1. **Offline: Label-Specific Subgraph Construction**
   Build an NSG (Navigable Small World Graph) index for each label partition, enabling focused traversal within relevant data subsets.

2. **Offline: Topology-Aware Entry Point Generation**
   Apply K-means clustering on each subgraph to partition nodes into clusters, then select entry points via out-degree-based hub refinement within each cluster.

3. **Online: Query-Centric Adaptive Search**
   At query time, select the entry point closest to the query vector and perform graph traversal on the corresponding label subgraph.


---

## Requirements

### System
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Compiler**: GCC 9+ with C++17 support
- **CMake**: 3.10+

### Libraries
| Library | Description | Source |
|---------|-------------|--------|
| EFANNA2E (NSG) | NSG graph construction | https://github.com/ZJULearning/nsg |
| efanna_graph | NN-Descent for KNN graph | https://github.com/ZJULearning/efanna_graph |
| FAISS | K-means clustering | https://github.com/facebookresearch/faiss |
| OpenBLAS | BLAS implementation | `apt install libopenblas-dev` |
| Boost | C++ utilities | `apt install libboost-all-dev` |
| OpenMP | Parallel processing | Included with GCC |

### Python
- Python 3.8+
- PyYAML (`pip install pyyaml`)

> For detailed environment setup, see `environment.yml`.

---

## Building the Project

### 1. Build External Libraries
```bash
# EFANNA2E (NSG)
git clone https://github.com/ZJULearning/nsg.git
cd nsg && mkdir build && cd build
cmake .. && make -j$(nproc)

# efanna_graph (NN-Descent)
git clone https://github.com/ZJULearning/efanna_graph.git
cd efanna_graph && mkdir build && cd build
cmake .. && make -j$(nproc)

# FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss && mkdir build && cd build
cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF
make -j$(nproc)
```

### 2. Build Q-LEAF
```bash
cd Q-LEAF
mkdir build && cd build

cmake .. \
    -DEFANNA2E_DIR=/path/to/nsg \
    -DEFANNA_GRAPH_DIR=/path/to/efanna_graph \
    -DFAISS_DIR=/path/to/faiss

make -j$(nproc)
```

---

## How to Reproduce

### Step 1: Data Preparation

Download datasets (see [Datasets](#datasets) section) and generate labels:
```bash
python scripts/generate_segmented_labels.py --dataset sift10m
```

Place data in the following structure:
```
data/
├── sift10m/
│   ├── sift10m_base.fvecs
│   ├── sift10m_query.fvecs
│   ├── base_labels.txt
│   └── query_labels.txt
└── ...
```

### Step 2: Build Subgraphs

Build label-specific NSG subgraphs:
```bash
# Using Python wrapper (recommended)
python scripts/build_and_generate_eps.py --dataset sift10m --task build

# Or direct execution
./build/bin/build_nsg_subgraphs \
    data/sift10m/sift10m_base.fvecs \
    data/sift10m/base_labels.txt \
    NSG_subgraph_output/sift10m \
    sift10m \
    40 50 8 10 100 \
    30 32 100
```
> Dataset-specific parameters (subgraph build, entry point) are defined in `config/*.yaml`.

### Step 3: Compute Entry Points

Generate topology-aware entry points using K-means clustering and out-degree-based hub refinement:
```bash
# Using Python wrapper
python scripts/build_and_generate_eps.py --dataset sift10m --num-eps 32

# Or direct execution
./build/bin/compute_ep_nsg \
    --base-fvecs data/sift10m/sift10m_base.fvecs \
    --graph-dir NSG_subgraph_output/sift10m/K40_L50_iter8_S10_R100_NSG_L30_R32_C100 \
    --output-dir NSG_subgraph_output/sift10m/K40_L50_iter8_S10_R100_NSG_L30_R32_C100/entry_points_32 \
    --num-clusters 32
```

### Step 4: Run Benchmarks

#### Q-LEAF
```bash
# Using Python wrapper (recommended)
python scripts/run_qLeaf_benchmark.py \
    --dataset sift10m \
    --num-labels 100 \
    --graph-config K40_L50_iter8_S10_R100_NSG_L30_R32_C100 \
    --num-eps 32 \
    --k 10 \
    --search-l 10,30,50,70,90

# Or direct execution
./build/bin/nsg_qleaf_single_ep \
    data/sift10m/sift10m_base.fvecs \
    NSG_subgraph_output/sift10m/K40_L50_iter8_S10_R100_NSG_L30_R32_C100/config.csv \
    data/sift10m/sift10m_query.fvecs \
    data/sift10m/query_labels.txt \
    10 \
    "50,100,150,200" \
    data/sift10m/groundtruth.ivecs
```

Results are saved to `results/` in CSV format (Recall@10, QPS).


---

## Datasets

| Dataset | Dimensions | Base Vectors | Queries | Download |
|---------|------------|--------------|---------|----------|
| SIFT10M | 128 | 10,000,000 | 1,000 | https://archive.ics.uci.edu/dataset/353/sift10m |
| Deep10M | 96 | 9,990,000 | 10,000 | https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search |
| Text2Image10M | 200 | 10,000,000 | 100,000 | https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search |
| YFCC10M | 192 | 10,000,000 | 100,000 | https://github.com/harsha-simhadri/big-ann-benchmarks/tree/main |

### File Formats

- **`.fvecs`**: Binary format for float vectors. Each vector: `[dim (4 bytes)] [float × dim]`
- **`.ivecs`**: Binary format for integer vectors (ground truth). Same structure as `.fvecs` but with integers.
- **`.txt`**: Text format for labels. One label per line, corresponding to each base vector.

---

## Baselines

We compare Q-LEAF against the following methods:

| Method | Reference | Repository |
|--------|-----------|------------|
| NHQ | Wang et al., NeurIPS 2023 | https://github.com/KGLab-HDU/TKDE-under-review-Native-Hybrid-Queries-via-ANNS |
| ACORN | Patel et al., SIGMOD 2024 | https://github.com/guestrin-lab/ACORN |
| Filtered-DiskANN | Gollapudi et al., 2023 | https://github.com/microsoft/DiskANN |
| UNG | Chen et al., 2025 | https://github.com/YZ-Cai/Unified-Navigating-Graph |

> Refer to the experimental section of our paper for the specific parameters used for each baseline.
