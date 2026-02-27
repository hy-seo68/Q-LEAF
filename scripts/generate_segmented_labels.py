#!/usr/bin/env python3
"""
Multi-Dataset Segmented Selectivity Labeling Script
Usage:
    python generate_segmented_labels.py --dataset sift10m
    python generate_segmented_labels.py --dataset deep10m
    python generate_segmented_labels.py --dataset text2image10m
    python generate_segmented_labels.py --dataset yfcc10m
"""

import numpy as np
import os
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct
import time
import multiprocessing as mp
from functools import partial

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS library not found. Ground Truth computation is not available.")
    FAISS_AVAILABLE = False

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_base_dir() -> Path:
    env_path = os.environ.get("QLEAF_BASE_DIR")
    if env_path:
        return Path(env_path)
    return get_project_root()

def get_dataset_dir() -> Path:
    env_path = os.environ.get("QLEAF_DATASET_DIR")
    if env_path:
        return Path(env_path)
    return get_project_root() / "data"


# Dataset Configuration
@dataclass
class DatasetConfig:
    name: str
    base_path: str
    query_path: str
    dimension: int
    num_base_vectors: int
    num_queries: int
    output_dir: str


def get_dataset_configs() -> Dict[str, DatasetConfig]:
    base_output_dir = str(get_base_dir() / "data")
    dataset_base = str(get_dataset_dir())

    configs = {
        "sift10m": DatasetConfig(
            name="SIFT10M",
            base_path=f"{dataset_base}/sift10m/sift10m_base.fvecs",
            query_path=f"{dataset_base}/sift10m/sift10m_query.fvecs",
            dimension=128,
            num_base_vectors=10_000_000,
            num_queries=1_000,
            output_dir=f"{base_output_dir}/sift10m"
        ),
        "deep10m": DatasetConfig(
            name="DEEP10M",
            base_path=f"{dataset_base}/deep10m/deep-image-96-angular_base.fvecs",
            query_path=f"{dataset_base}/deep10m/deep-image-96-angular_query.fvecs",
            dimension=96,
            num_base_vectors=9_990_000,
            num_queries=10_000,
            output_dir=f"{base_output_dir}/deep10m"
        ),
        "text2image10m": DatasetConfig(
            name="TEXT2IMAGE10M",
            base_path=f"{dataset_base}/text2image10m/base.10M.fvecs",
            query_path=f"{dataset_base}/text2image10m/query.public.100K.fvecs",
            dimension=200,
            num_base_vectors=10_000_000,
            num_queries=100_000,
            output_dir=f"{base_output_dir}/text2image10m"
        ),
        "yfcc10m": DatasetConfig(
            name="YFCC10M",
            base_path=f"{dataset_base}/yfcc10M/base.10M.fvecs",
            query_path=f"{dataset_base}/yfcc10M/query.public.100K.fvecs",
            dimension=192,
            num_base_vectors=10_000_000,
            num_queries=100_000,
            output_dir=f"{base_output_dir}/yfcc10m"
        )
    }
    return configs

BATCH_SIZE = 1_000_000          
RANDOM_SEED = 42                
K_NEIGHBORS = 100               
MIN_VECTORS_PER_LABEL = 1000    

# Query allocation ratios (SIFT baseline: 20%, 30%, 30%, 20%)
QUERY_ALLOCATION_RATIOS = {
    "10%_Large": 0.20,
    "1%_Mid": 0.30,
    "0.1%_Low": 0.30,
    "0.01%_Sparse": 0.20
}

# Segment selectivity configuration template (number of labels and target selectivity)
SELECTIVITY_SEGMENTS_TEMPLATE = {
    "10%_Large": {
        "num_labels": 5,
        "target_selectivity": 0.10,
        "description": "Large subgraphs (~1M vectors each)"
    },
    "1%_Mid": {
        "num_labels": 40,
        "target_selectivity": 0.01,
        "description": "Medium subgraphs (~100K vectors each)"
    },
    "0.1%_Low": {
        "num_labels": 90,
        "target_selectivity": 0.001,
        "description": "Small subgraphs (~10K vectors each)"
    },
    "0.01%_Sparse": {
        "num_labels": 100,
        "target_selectivity": 0.0001,
        "description": "Sparse subgraphs (~1K vectors each)"
    }
}


def compute_query_allocation(total_queries: int) -> Dict[str, int]:
    """
    Compute query allocation per segment based on total number of queries
    """
    segments_order = ["10%_Large", "1%_Mid", "0.1%_Low", "0.01%_Sparse"]
    allocation = {}
    remaining = total_queries

    for i, segment in enumerate(segments_order):
        if i == len(segments_order) - 1:
            allocation[segment] = remaining
        else:
            ratio = QUERY_ALLOCATION_RATIOS[segment]
            num_queries = int(total_queries * ratio)
            allocation[segment] = num_queries
            remaining -= num_queries

    # Verify minimum 2 queries per label for 0.01% Sparse segment
    sparse_labels = SELECTIVITY_SEGMENTS_TEMPLATE["0.01%_Sparse"]["num_labels"]
    min_sparse_queries = sparse_labels * 2

    if allocation["0.01%_Sparse"] < min_sparse_queries:
        print(f"  Warning: Number of queries in 0.01%_Sparse segment ({allocation['0.01%_Sparse']}) "
              f"is less than minimum required ({min_sparse_queries}).")
        print(f"           Adjust number of labels or modify query allocation ratios.")

    return allocation


def generate_selectivity_segments(total_queries: int) -> Dict[str, dict]:
    """
    Generate segment configuration based on total number of queries
    """
    query_allocation = compute_query_allocation(total_queries)

    segments = {}
    for segment_name, template in SELECTIVITY_SEGMENTS_TEMPLATE.items():
        segments[segment_name] = {
            "num_labels": template["num_labels"],
            "target_selectivity": template["target_selectivity"],
            "num_queries": query_allocation[segment_name],
            "description": template["description"]
        }

    return segments


def get_fvecs_info(filepath: str) -> Tuple[int, int]:
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    vector_size = 4 + dim * 4
    num_vectors = file_size // vector_size
    return num_vectors, dim


def read_fvecs_mmap(filepath: str) -> np.ndarray:
    num_vectors, dim = get_fvecs_info(filepath)

    data = np.memmap(filepath, dtype='float32', mode='r')
    vectors_per_row = dim + 1
    data = data.reshape(-1, vectors_per_row)
    vectors = data[:, 1:].copy().astype(np.float32)

    return vectors


def write_ivecs(filepath: str, data: np.ndarray):
    data = np.asarray(data, dtype=np.int32)
    num_queries, k = data.shape

    with open(filepath, 'wb') as f:
        for i in range(num_queries):
            f.write(struct.pack('i', k))
            f.write(data[i].tobytes())

    print(f"  .ivecs file saved: {filepath}")
    print(f"    - Number of queries: {num_queries:,}, k: {k}")



def generate_segmented_probabilities(config: DatasetConfig,
                                      selectivity_segments: Dict[str, dict]
                                     ) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate label probability distribution based on segment selectivity
    """
    probabilities = []
    label_info = []

    label_id = 0
    total_probability = 0.0

    print("\nGenerating labels per segment:")
    print("="*80)

    for segment_name in ["10%_Large", "1%_Mid", "0.1%_Low", "0.01%_Sparse"]:
        seg_config = selectivity_segments[segment_name]
        num_labels = seg_config["num_labels"]
        target_sel = seg_config["target_selectivity"]

        for _ in range(num_labels):
            probabilities.append(target_sel)
            label_info.append({
                "label_id": label_id,
                "segment": segment_name,
                "target_selectivity": target_sel
            })
            label_id += 1

        segment_prob = num_labels * target_sel
        total_probability += segment_prob

        expected_vectors = int(config.num_base_vectors * target_sel)
        print(f"  {segment_name}:")
        print(f"    - Number of labels: {num_labels}")
        print(f"    - Target selectivity: {target_sel*100:.4f}%")
        print(f"    - Expected vectors/label: {expected_vectors:,}")
        print(f"    - Allocated queries: {seg_config['num_queries']:,}")
        print(f"    - Queries/label: {seg_config['num_queries']/num_labels:.2f}")
        print(f"    - Segment probability sum: {segment_prob:.4f}")

    print("-"*80)
    print(f"  Total number of labels: {label_id}")
    print(f"  Probability sum (before normalization): {total_probability:.6f}")

    # Normalization
    probabilities = np.array(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    print(f"  Probability sum (after normalization): {probabilities.sum():.6f}")
    print("="*80)

    return probabilities, label_info


def assign_labels_to_base_vectors(num_vectors: int,
                                   probabilities: np.ndarray,
                                   batch_size: int = BATCH_SIZE,
                                   seed: int = RANDOM_SEED) -> np.ndarray:
    """Assign labels to base vectors in batches"""
    np.random.seed(seed)
    labels = np.empty(num_vectors, dtype=np.int32)
    num_labels = len(probabilities)
    num_batches = (num_vectors + batch_size - 1) // batch_size

    print(f"\nStarting base vector label assignment")
    print(f"  Total vectors: {num_vectors:,}")
    print(f"  Total labels: {num_labels}")
    print(f"  Batch size: {batch_size:,}")

    start_time = time.time()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_vectors)
        current_batch_size = end_idx - start_idx
        batch_labels = np.random.choice(num_labels, size=current_batch_size, p=probabilities)
        labels[start_idx:end_idx] = batch_labels

        if (i + 1) % 5 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - start_time
            print(f"  Progress: {end_idx:,}/{num_vectors:,} ({100*end_idx/num_vectors:.1f}%) - {elapsed:.1f}s")

    return labels


def compute_label_statistics(labels: np.ndarray,
                              label_info: List[dict],
                              num_vectors: int
                             ) -> Tuple[Counter, Dict[str, dict]]:
    """Compute label distribution statistics"""
    label_counts = Counter(labels)

    segment_stats = defaultdict(lambda: {
        "labels": [],
        "total_vectors": 0,
        "target_selectivity": 0
    })

    for info in label_info:
        label_id = info["label_id"]
        segment = info["segment"]
        target_sel = info["target_selectivity"]
        actual_count = label_counts.get(label_id, 0)
        actual_sel = actual_count / num_vectors

        segment_stats[segment]["labels"].append({
            "label_id": label_id,
            "count": actual_count,
            "actual_selectivity": actual_sel,
            "target_selectivity": target_sel
        })
        segment_stats[segment]["total_vectors"] += actual_count
        segment_stats[segment]["target_selectivity"] = target_sel

    return label_counts, segment_stats


def print_segment_report(segment_stats: Dict[str, dict],
                          num_vectors: int,
                          selectivity_segments: Dict[str, dict]) -> bool:
    print("\n" + "="*100)
    print("Segment Label Distribution Report")
    print("="*100)

    print("\n[Summary Table]")
    print("-"*100)
    header = f"{'Segment':<15} {'Labels':>8} {'Target Sel.':>12} {'Actual Avg':>12} {'Std Dev':>10} {'Avg Vectors':>12} {'Queries':>10} {'Q/Label':>10}"
    print(header)
    print("-"*100)

    all_min_vectors = float('inf')
    problem_labels = []
    segments_order = ["10%_Large", "1%_Mid", "0.1%_Low", "0.01%_Sparse"]

    for segment_name in segments_order:
        stats = segment_stats[segment_name]
        seg_config = selectivity_segments[segment_name]
        num_labels = len(stats["labels"])
        target_sel = stats["target_selectivity"]
        allocated_queries = seg_config["num_queries"]

        if num_labels > 0:
            actual_sels = [l["actual_selectivity"] for l in stats["labels"]]
            avg_actual_sel = np.mean(actual_sels)
            std_actual_sel = np.std(actual_sels)
            counts = [l["count"] for l in stats["labels"]]
            avg_vectors = np.mean(counts)
            min_vectors = min(counts)
            max_vectors = max(counts)

            all_min_vectors = min(all_min_vectors, min_vectors)

            for l in stats["labels"]:
                if l["count"] < MIN_VECTORS_PER_LABEL:
                    problem_labels.append((l["label_id"], segment_name, l["count"]))

            queries_per_label = allocated_queries / num_labels

            print(f"{segment_name:<15} {num_labels:>8} {target_sel*100:>11.4f}% "
                  f"{avg_actual_sel*100:>11.4f}% {std_actual_sel*100:>9.6f}% "
                  f"{avg_vectors:>12,.0f} {allocated_queries:>10,} {queries_per_label:>10.2f}")
        else:
            print(f"{segment_name:<15} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

    print("-"*100)

    print("\n[Detailed Statistics]")
    for segment_name in segments_order:
        stats = segment_stats[segment_name]
        num_labels = len(stats["labels"])

        if num_labels > 0:
            counts = [l["count"] for l in stats["labels"]]
            actual_sels = [l["actual_selectivity"] for l in stats["labels"]]

            min_count = min(counts)
            max_count = max(counts)
            std_count = np.std(counts)
            median_count = np.median(counts)

            min_sel = min(actual_sels)
            max_sel = max(actual_sels)

            print(f"\n  {segment_name}:")
            print(f"    - Label range: {stats['labels'][0]['label_id']} ~ {stats['labels'][-1]['label_id']}")
            print(f"    - Vector count:")
            print(f"        min={min_count:,}, max={max_count:,}, median={median_count:,.0f}, std={std_count:,.0f}")
            print(f"    - Selectivity range:")
            print(f"        min={min_sel*100:.6f}%, max={max_sel*100:.6f}%")
            print(f"    - Total vectors: {stats['total_vectors']:,} ({stats['total_vectors']/num_vectors*100:.2f}%)")

    # Minimum vector count validation
    print("\n" + "-"*100)
    print(f"[Minimum Vector Count Validation]")
    print(f"  Minimum vectors across all labels: {all_min_vectors:,}")
    print(f"  Required minimum vectors: {MIN_VECTORS_PER_LABEL:,}")

    if problem_labels:
        print(f"\n  Warning: {len(problem_labels)} labels below minimum vector count!")
        for label_id, segment, count in problem_labels[:10]:
            print(f"    - Label {label_id} ({segment}): {count:,}")
        if len(problem_labels) > 10:
            print(f"    ... and {len(problem_labels)-10} more")
    else:
        print(f"  All labels have at least {MIN_VECTORS_PER_LABEL:,} vectors.")

    print("="*100)

    return len(problem_labels) == 0


def assign_labels_to_queries(num_queries: int,
                              segment_stats: Dict[str, dict],
                              selectivity_segments: Dict[str, dict],
                              seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Assign labels to query vectors with specified number of queries per selectivity group
    """
    query_labels = np.empty(num_queries, dtype=np.int32)
    segments_order = ["10%_Large", "1%_Mid", "0.1%_Low", "0.01%_Sparse"]

    # Get query allocation from configuration
    total_configured_queries = sum(selectivity_segments[seg]["num_queries"] for seg in segments_order)

    print(f"\nQuery label assignment")
    print(f"  Total queries: {num_queries:,}")
    print(f"  Configured query total: {total_configured_queries:,}")

    if num_queries != total_configured_queries:
        print(f"  Warning: Actual query count ({num_queries:,}) differs from configured total ({total_configured_queries:,}).")
        print(f"           Adjusting proportionally.")

    current_idx = 0
    np.random.seed(seed + 1)  # Separate seed for queries

    print("\n  Query allocation per segment:")
    print("  " + "-"*70)

    for segment_name in segments_order:
        config = selectivity_segments[segment_name]
        group_size = int(num_queries * config["num_queries"] / total_configured_queries)

        if segment_name == segments_order[-1]:
            group_size = num_queries - current_idx

        stats = segment_stats[segment_name]

        if stats["labels"]:
            available_labels = [l["label_id"] for l in stats["labels"]]

            for j in range(group_size):
                label = available_labels[j % len(available_labels)]
                query_labels[current_idx] = label
                current_idx += 1

            queries_per_label = group_size / len(available_labels)
            min_queries_per_label = group_size // len(available_labels)
            max_queries_per_label = min_queries_per_label + (1 if group_size % len(available_labels) > 0 else 0)

            print(f"  {segment_name:<15}: {group_size:>8,} queries -> {len(available_labels):>3} labels")
            print(f"                     queries/label: avg {queries_per_label:.2f}, range [{min_queries_per_label}-{max_queries_per_label}]")
        else:
            print(f"  {segment_name}: No labels! Using default value (0)")
            query_labels[current_idx:current_idx + group_size] = 0
            current_idx += group_size

    print("  " + "-"*70)

    return query_labels


def save_labels(labels: np.ndarray, filepath: str, num_attributes: int = 1):
    print(f"\nSaving labels: {filepath}")
    num_vectors = len(labels)
    with open(filepath, 'w') as f:
        f.write(f"{num_vectors} {num_attributes}\n")
        for label in labels:
            f.write(f"{label}\n")
    print(f"  Done: {num_vectors:,} labels saved (header: {num_vectors} {num_attributes})")


def build_label_to_indices_map(base_labels: np.ndarray) -> Dict[int, np.ndarray]:
    print("\nGrouping indices by label...")
    start_time = time.time()

    label_to_indices = defaultdict(list)

    for global_idx, label in enumerate(base_labels):
        label_to_indices[label].append(global_idx)

    for label in label_to_indices:
        label_to_indices[label] = np.array(label_to_indices[label], dtype=np.int64)

    elapsed = time.time() - start_time
    print(f"  Grouping complete: {len(label_to_indices)} label groups ({elapsed:.2f}s)")

    return label_to_indices


def compute_gt_for_label_batch(label_query_pairs: List[Tuple[int, List[int]]],
                                base_vectors: np.ndarray,
                                query_vectors: np.ndarray,
                                label_to_indices: Dict[int, np.ndarray],
                                k: int) -> Dict[int, np.ndarray]:
    results = {}

    for label, query_indices in label_query_pairs:
        base_indices = label_to_indices.get(label, np.array([], dtype=np.int64))

        if len(base_indices) == 0:
            for q_idx in query_indices:
                results[q_idx] = np.full(k, -1, dtype=np.int32)
            continue

        label_base_vectors = base_vectors[base_indices].astype(np.float32)
        dim = label_base_vectors.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(label_base_vectors)

        label_query_vectors = query_vectors[query_indices].astype(np.float32)
        actual_k = min(k, len(base_indices))

        _, local_indices = index.search(label_query_vectors, actual_k)

        for i, q_idx in enumerate(query_indices):
            gt_row = np.full(k, -1, dtype=np.int32)
            for j in range(actual_k):
                local_idx = local_indices[i, j]
                if local_idx >= 0:
                    gt_row[j] = base_indices[local_idx]
            results[q_idx] = gt_row

    return results


def compute_ground_truth_with_faiss(query_vectors: np.ndarray,
                                     query_labels: np.ndarray,
                                     base_vectors: np.ndarray,
                                     label_to_indices: Dict[int, np.ndarray],
                                     k: int = K_NEIGHBORS,
                                     num_workers: int = None) -> Tuple[np.ndarray, List[int]]:
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS library is not installed.")

    num_queries = len(query_vectors)
    ground_truth = np.full((num_queries, k), -1, dtype=np.int32)

    # Group queries by label
    query_by_label = defaultdict(list)
    for q_idx, label in enumerate(query_labels):
        query_by_label[label].append(q_idx)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Maximum 8 workers

    print(f"\nStarting Ground Truth computation (k={k})")
    print(f"  Label groups to process: {len(query_by_label)}")
    print(f"  Total queries: {num_queries:,}")

    total_queries_processed = 0
    start_time = time.time()
    insufficient_neighbors = []

    for label, query_indices in query_by_label.items():
        label_start = time.time()
        base_indices = label_to_indices.get(label, np.array([], dtype=np.int64))

        if len(base_indices) == 0:
            print(f"  Warning: Label {label}: No base vectors!")
            insufficient_neighbors.extend(query_indices)
            continue

        label_base_vectors = base_vectors[base_indices].astype(np.float32)
        dim = label_base_vectors.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(label_base_vectors)

        label_query_vectors = query_vectors[query_indices].astype(np.float32)
        actual_k = min(k, len(base_indices))

        _, local_indices = index.search(label_query_vectors, actual_k)

        for i, q_idx in enumerate(query_indices):
            for j in range(actual_k):
                local_idx = local_indices[i, j]
                if local_idx >= 0:
                    ground_truth[q_idx, j] = base_indices[local_idx]

            if actual_k < k:
                insufficient_neighbors.append(q_idx)

        total_queries_processed += len(query_indices)
        label_time = time.time() - label_start

        # Adaptive progress output
        if len(query_by_label) <= 50:
            print(f"  Label {label:>3}: {len(base_indices):>10,} vectors, "
                  f"{len(query_indices):>6,} queries, {label_time:.2f}s")
        elif total_queries_processed % max(1, num_queries // 20) < len(query_indices):
            elapsed = time.time() - start_time
            progress = 100 * total_queries_processed / num_queries
            print(f"  Progress: {total_queries_processed:,}/{num_queries:,} ({progress:.1f}%) - {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nGround Truth computation complete")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Processed queries: {total_queries_processed:,}")
    print(f"  Processing rate: {total_queries_processed/total_time:.1f} queries/s")

    return ground_truth, insufficient_neighbors


def validate_ground_truth(ground_truth: np.ndarray,
                           query_labels: np.ndarray,
                           selectivity_segments: Dict[str, dict]) -> bool:
    """Ground Truth validation (enhanced version)"""
    print("\n" + "="*100)
    print("Ground Truth Validation Report")
    print("="*100)

    num_queries, k = ground_truth.shape

    # 1. Overall validity check
    invalid_mask = ground_truth == -1
    invalid_per_query = np.sum(invalid_mask, axis=1)
    fully_valid_queries = np.sum(invalid_per_query == 0)
    partially_valid_queries = np.sum((invalid_per_query > 0) & (invalid_per_query < k))
    fully_invalid_queries = np.sum(invalid_per_query == k)

    print(f"\n[Overall Validity Check]")
    print(f"  Total queries: {num_queries:,}")
    print(f"  k (neighbors): {k}")
    print(f"  Fully valid queries (all {k} neighbors): {fully_valid_queries:,} ({100*fully_valid_queries/num_queries:.2f}%)")
    print(f"  Partially valid queries (some -1): {partially_valid_queries:,} ({100*partially_valid_queries/num_queries:.2f}%)")
    print(f"  Fully invalid queries (all -1): {fully_invalid_queries:,} ({100*fully_invalid_queries/num_queries:.2f}%)")

    # 2. Per-segment validation
    print(f"\n[Per-Segment Validation]")
    segments_order = ["10%_Large", "1%_Mid", "0.1%_Low", "0.01%_Sparse"]
    total_configured_queries = sum(selectivity_segments[seg]["num_queries"] for seg in segments_order)

    print("-"*100)
    header = f"{'Segment':<15} {'Queries':>10} {'Fully Valid':>12} {'Valid Rate':>10} {'Avg Valid Neighbors':>20} {'Min Valid Neighbors':>20}"
    print(header)
    print("-"*100)

    all_valid = True
    current_idx = 0

    for segment_name in segments_order:
        config = selectivity_segments[segment_name]
        group_size = int(num_queries * config["num_queries"] / total_configured_queries)
        if segment_name == segments_order[-1]:
            group_size = num_queries - current_idx

        segment_gt = ground_truth[current_idx:current_idx + group_size]
        segment_invalid = segment_gt == -1
        valid_per_query = k - np.sum(segment_invalid, axis=1)

        fully_valid = np.sum(valid_per_query == k)
        avg_valid = np.mean(valid_per_query)
        min_valid = np.min(valid_per_query)
        valid_rate = 100 * fully_valid / group_size if group_size > 0 else 0

        if min_valid < k:
            all_valid = False

        print(f"{segment_name:<15} {group_size:>10,} {fully_valid:>12,} {valid_rate:>9.1f}% {avg_valid:>15.1f} {min_valid:>15}")
        current_idx += group_size

    print("-"*100)

    # 3. Final validation result
    if all_valid and fully_invalid_queries == 0:
        print(f"\nGround Truth validation passed: All queries have valid neighbors.")
        return True
    else:
        print(f"\nGround Truth validation warning:")
        if fully_invalid_queries > 0:
            print(f"  - {fully_invalid_queries:,} queries are fully invalid.")
        if partially_valid_queries > 0:
            print(f"  - {partially_valid_queries:,} queries have fewer than {k} neighbors.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Segmented Selectivity Labeling Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sift10m", "deep10m", "text2image10m", "yfcc10m"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_NEIGHBORS,
        help=f"Number of neighbors for Ground Truth (default: {K_NEIGHBORS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip Ground Truth computation (generate labels only)"
    )

    args = parser.parse_args()

    # Load dataset configuration
    configs = get_dataset_configs()
    config = configs[args.dataset]

    print("="*100)
    print(f"{config.name} Segmented Selectivity Labeling")
    print("="*100)

    print(f"\n[Dataset Configuration]")
    print(f"  - Dataset: {config.name}")
    print(f"  - Base vectors: {config.num_base_vectors:,}")
    print(f"  - Dimension: {config.dimension}")
    print(f"  - Number of queries: {config.num_queries:,}")
    print(f"  - Base file: {config.base_path}")
    print(f"  - Query file: {config.query_path}")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Ground Truth k: {args.k}")
    print(f"  - Random seed: {args.seed}")

    os.makedirs(config.output_dir, exist_ok=True)
    print(f"\nOutput directory created/verified: {config.output_dir}")

    if not os.path.exists(config.base_path):
        print(f"\nError: Base file not found: {config.base_path}")
        return
    if not os.path.exists(config.query_path):
        print(f"\nError: Query file not found: {config.query_path}")
        return

    actual_num_queries, actual_dim = get_fvecs_info(config.query_path)
    print(f"\n[File Validation]")
    print(f"  - Actual query count in file: {actual_num_queries:,}")
    print(f"  - Actual dimension in file: {actual_dim}")

    if actual_dim != config.dimension:
        print(f"  Warning: Dimension differs from config ({config.dimension}). Using actual value ({actual_dim}).")

    selectivity_segments = generate_selectivity_segments(actual_num_queries)

    print(f"\n[1] Generating segment probability distribution")
    probabilities, label_info = generate_segmented_probabilities(config, selectivity_segments)
    total_labels = len(probabilities)
    print(f"\nTotal labels generated: {total_labels}")

    print(f"\n[2] Assigning labels to base vectors")
    base_labels = assign_labels_to_base_vectors(
        config.num_base_vectors, probabilities, seed=args.seed
    )
    print(f"\n[3] Computing label statistics")
    label_counts, segment_stats = compute_label_statistics(
        base_labels, label_info, config.num_base_vectors
    )

    validation_passed = print_segment_report(
        segment_stats, config.num_base_vectors, selectivity_segments
    )

    if not validation_passed:
        print("\nWarning: Some labels do not meet minimum vector count.")
        print("  Graph building may encounter issues.")

    print(f"\n[4] Saving base labels")
    base_labels_path = os.path.join(config.output_dir, "base_labels_for_selectivity.txt")
    save_labels(base_labels, base_labels_path)

    print(f"\n[5] Assigning labels to query vectors (query count: {actual_num_queries:,})")
    query_labels = assign_labels_to_queries(
        actual_num_queries, segment_stats, selectivity_segments, seed=args.seed
    )

    print(f"\n[6] Saving query labels")
    query_labels_path = os.path.join(config.output_dir, "query_labels_for_selectivity.txt")
    save_labels(query_labels, query_labels_path)

    print("\n" + "="*100)
    print("Query Label Statistics")
    print("="*100)
    query_label_counts = Counter(query_labels)
    print(f"Unique labels used in queries: {len(query_label_counts)}")

    segments_order = ["10%_Large", "1%_Mid", "0.1%_Low", "0.01%_Sparse"]
    total_configured_queries = sum(selectivity_segments[seg]["num_queries"] for seg in segments_order)

    print(f"\n[Query Distribution per Segment - Details]")
    print("-"*80)
    print(f"{'Segment':<15} {'Queries':>10} {'Unique Labels':>15} {'Q/Label':>12} {'Q/Label Range':>20}")
    print("-"*80)

    current_idx = 0
    for segment_name in segments_order:
        seg_config = selectivity_segments[segment_name]
        group_size = int(actual_num_queries * seg_config["num_queries"] / total_configured_queries)
        if segment_name == segments_order[-1]:
            group_size = actual_num_queries - current_idx

        segment_queries = query_labels[current_idx:current_idx + group_size]
        unique_labels = len(set(segment_queries))
        queries_per_label = group_size / seg_config["num_labels"]


        label_query_counts = Counter(segment_queries)
        min_q = min(label_query_counts.values())
        max_q = max(label_query_counts.values())

        print(f"{segment_name:<15} {group_size:>10,} {unique_labels:>10} {queries_per_label:>12.2f} [{min_q:>5} - {max_q:>5}]")
        current_idx += group_size

    print("-"*80)

    # Ground Truth computation (using FAISS)
    if not args.skip_gt and FAISS_AVAILABLE:
        print("\n" + "="*100)
        print("[7] Ground Truth Computation (using FAISS)")
        print("="*100)

        print("\nLoading base vectors (mmap)...")
        load_start = time.time()
        base_vectors = read_fvecs_mmap(config.base_path)
        print(f"  Load complete: {base_vectors.shape}, dtype={base_vectors.dtype}")
        print(f"  Time elapsed: {time.time()-load_start:.2f}s")

        print("\nLoading query vectors...")
        load_start = time.time()
        query_vectors = read_fvecs_mmap(config.query_path)
        print(f"  Load complete: {query_vectors.shape}, dtype={query_vectors.dtype}")
        print(f"  Time elapsed: {time.time()-load_start:.2f}s")

        label_to_indices = build_label_to_indices_map(base_labels)

        ground_truth, insufficient_neighbors = compute_ground_truth_with_faiss(
            query_vectors, query_labels, base_vectors,
            label_to_indices, k=args.k
        )

        print(f"\n[8] Saving Ground Truth")
        gt_path = os.path.join(config.output_dir, "groundtruth_selectivity.ivecs")
        write_ivecs(gt_path, ground_truth)

        print(f"\n[9] Validating Ground Truth")
        validate_ground_truth(ground_truth, query_labels, selectivity_segments)

    elif args.skip_gt:
        print("\n[7-9] Skipping Ground Truth computation (--skip-gt option)")
    else:
        print("\nWarning: FAISS is not installed. Skipping Ground Truth computation.")

    print("\n" + "="*100)
    print("Complete!")
    print("="*100)
    print(f"\nGenerated files:")
    print(f"  - Base labels: {base_labels_path}")
    print(f"  - Query labels: {query_labels_path}")
    if not args.skip_gt and FAISS_AVAILABLE:
        print(f"  - Ground Truth: {os.path.join(config.output_dir, 'groundtruth_selectivity.ivecs')}")

    print(f"\n[Label Summary]")
    print(f"  - Total labels: {total_labels}")
    print(f"  - Min vectors/label: {min(label_counts.values()):,}")
    print(f"  - Max vectors/label: {max(label_counts.values()):,}")

    # Save configuration file
    config_path = os.path.join(config.output_dir, "labeling_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"# Segmented Selectivity Labeling Configuration\n")
        f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"dataset: {config.name}\n")
        f.write(f"base_path: {config.base_path}\n")
        f.write(f"query_path: {config.query_path}\n")
        f.write(f"dimension: {actual_dim}\n")
        f.write(f"num_base_vectors: {config.num_base_vectors}\n")
        f.write(f"num_queries: {actual_num_queries}\n")
        f.write(f"k_neighbors: {args.k}\n")
        f.write(f"random_seed: {args.seed}\n")
        f.write(f"total_labels: {total_labels}\n\n")

        f.write(f"# Selectivity Segments\n")
        for segment_name in segments_order:
            seg_config = selectivity_segments[segment_name]
            f.write(f"\n[{segment_name}]\n")
            f.write(f"  num_labels: {seg_config['num_labels']}\n")
            f.write(f"  target_selectivity: {seg_config['target_selectivity']}\n")
            f.write(f"  num_queries: {seg_config['num_queries']}\n")

    print(f"  - Configuration file: {config_path}")


if __name__ == "__main__":
    main()
