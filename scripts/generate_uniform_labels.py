#!/usr/bin/env python3
"""
Q-LEAF Label Dataset Generator

Generates uniform label distribution for filtered ANNS benchmarks.
Usage:
    python create_label_dataset.py --dataset sift10m --num-labels 12
    python create_label_dataset.py --dataset deep10m --num-labels 100
"""

import numpy as np
import struct
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm


def get_project_root() -> Path:
    """Get Q-LEAF project root directory"""
    return Path(__file__).resolve().parent.parent


def get_dataset_dir() -> Path:
    """Get dataset directory (can be overridden by environment variable)"""
    env_path = os.environ.get("QLEAF_DATASET_DIR")
    if env_path:
        return Path(env_path)
    return get_project_root() / "data"


# Dataset configurations
DATASET_CONFIGS = {
    "sift10m": {
        "base_file": "sift10m_base.fvecs",
        "query_file": "sift10m_query.fvecs",
        "num_base": 10_000_000,
        "num_queries": 1_000,
        "dimension": 128,
    },
    "deep10m": {
        "base_file": "deep-image-96-angular_base.fvecs",
        "query_file": "deep-image-96-angular_query.fvecs",
        "num_base": 9_990_000,
        "num_queries": 10_000,
        "dimension": 96,
    },
    "text2image10m": {
        "base_file": "base.10M.fvecs",
        "query_file": "query.public.100K.fvecs",
        "num_base": 10_000_000,
        "num_queries": 100_000,
        "dimension": 200,
    },
    "yfcc10m": {
        "base_file": "base.10M.fvecs",
        "query_file": "query.public.100K.fvecs",
        "num_base": 10_000_000,
        "num_queries": 100_000,
        "dimension": 192,
    },
}

# Default parameters
DEFAULT_NUM_LABELS = 12
DEFAULT_K = 10
DEFAULT_SEED = 42


def get_fvecs_info(filepath: str) -> Tuple[int, int]:
    """Get number of vectors and dimension from fvecs file"""
    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    vector_size = 4 + dim * 4
    num_vectors = file_size // vector_size
    return num_vectors, dim


def read_fvecs_mmap(filepath: str, max_count: int = None) -> np.ndarray:
    """Read fvecs file using memory mapping for efficiency"""
    print(f"Reading {filepath}...")
    num_vectors, dim = get_fvecs_info(filepath)

    if max_count and max_count < num_vectors:
        num_vectors = max_count

    data = np.memmap(filepath, dtype='float32', mode='r')
    vectors_per_row = dim + 1
    data = data.reshape(-1, vectors_per_row)

    if max_count:
        vectors = data[:max_count, 1:].copy().astype(np.float32)
    else:
        vectors = data[:, 1:].copy().astype(np.float32)

    print(f"Finished reading {len(vectors)} vectors, dim={dim}")
    return vectors


def generate_labels(size, num_labels, seed=42):
    """Generate uniformly distributed labels for dataset
    """
    print(f"  Generating labels with uniform distribution...")
    print(f"    - Total classes: {num_labels} (0~{num_labels-1})")
    print(f"    - Distribution method: Uniform distribution")

    np.random.seed(seed)

    # Generate uniformly distributed labels
    labels = np.array([i % num_labels for i in range(size)], dtype=np.int32)
    np.random.shuffle(labels)

    # Show label distribution statistics
    unique, counts = np.unique(labels, return_counts=True)
    print(f"    - Min count: {counts.min():,}")
    print(f"    - Max count: {counts.max():,}")
    print(f"    - Mean count: {counts.mean():.2f}")
    print(f"    - Std dev: {counts.std():.2f}")

    return labels


def save_labels(labels: np.ndarray, filepath: str, num_attributes: int = 1):
    """Save labels to text file with header (Q-LEAF format)

    Format:
        First line: <num_vectors> <num_attributes>
        Following lines: one label per line
    """
    print(f"Saving labels to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    num_vectors = len(labels)

    with open(filepath, 'w') as f:
        f.write(f"{num_vectors} {num_attributes}\n")
        for label in labels:
            f.write(f"{label}\n")

    print(f"Saved {num_vectors:,} labels (header: {num_vectors} {num_attributes})")


def compute_distance(query, base_vectors):
    diff = base_vectors - query
    distances_squared = np.sum(diff ** 2, axis=1)
    return distances_squared


def compute_label_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k=100):
    """Compute groundtruth for labeled filtered search
    """
    print(f"\nComputing label-filtered groundtruth (k={k})...")
    num_queries = len(query_vectors)
    groundtruth = np.zeros((num_queries, k), dtype=np.int32)

    # Group base vectors by label for efficient filtering
    print("Building label index...")
    label_to_indices = {}
    for label in np.unique(base_labels):
        label_to_indices[label] = np.where(base_labels == label)[0]

    print(f"\nLabel distribution in base:")
    for label in sorted(label_to_indices.keys()):
        print(f"  Label {label}: {len(label_to_indices[label]):,} vectors")

    print(f"\nProcessing each query for label-constrained KNN...")
    print(f"  - Query count: {num_queries:,}")
    print(f"  - K: {k}")

    for q_idx in tqdm(range(num_queries), desc="Processing queries"):
        query_vec = query_vectors[q_idx]
        query_label = query_labels[q_idx]

        # Get candidate indices with matching label
        if query_label not in label_to_indices:
            print(f"\nWarning: Query {q_idx} with label {query_label} has no matching base vectors!")
            groundtruth[q_idx] = -1
            continue

        candidate_indices = label_to_indices[query_label]

        if len(candidate_indices) < k:
            print(f"\nWarning: Query {q_idx} label {query_label} has only {len(candidate_indices)} vectors (k={k})")
            actual_k = len(candidate_indices)
        else:
            actual_k = k

        candidate_vectors = base_vectors[candidate_indices]

        distances = compute_distance(query_vec, candidate_vectors)

        nearest_indices = np.argsort(distances)[:actual_k]

        groundtruth[q_idx, :actual_k] = candidate_indices[nearest_indices]

        if actual_k < k:
            groundtruth[q_idx, actual_k:] = -1

    print("Groundtruth computation completed")
    return groundtruth


def save_ivecs(data: np.ndarray, filepath: str):
    """Save data in ivecs format"""
    print(f"Saving groundtruth to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = np.asarray(data, dtype=np.int32)
    num_queries, k = data.shape

    with open(filepath, 'wb') as f:
        for i in range(num_queries):
            f.write(struct.pack('i', k))
            f.write(data[i].tobytes())

    print(f"Saved groundtruth for {num_queries:,} queries (k={k})")


def main():
    parser = argparse.ArgumentParser(
        description="Q-LEAF Uniform Label Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_label_dataset.py --dataset sift10m --num-labels 12
    python create_label_dataset.py --dataset deep10m --num-labels 100 --k 100
    python create_label_dataset.py --dataset sift10m --num-labels 50 --skip-gt
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=DEFAULT_NUM_LABELS,
        help=f"Number of label classes (default: {DEFAULT_NUM_LABELS})"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of neighbors for groundtruth (default: {DEFAULT_K})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip groundtruth computation (generate labels only)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Custom suffix for output directory (default: label{num_labels})"
    )

    args = parser.parse_args()

    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    dataset_dir = get_dataset_dir()

    # Set paths
    base_fvecs = dataset_dir / args.dataset / config["base_file"]
    query_fvecs = dataset_dir / args.dataset / config["query_file"]

    # Output directory
    suffix = args.output_suffix if args.output_suffix else f"label{args.num_labels}"
    output_dir = dataset_dir / f"{args.dataset}_{suffix}"

    print("=" * 80)
    print(f"Q-LEAF Uniform Label Dataset Generator")
    print("=" * 80)
    print(f"\n[Configuration]")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of labels: {args.num_labels} (0 to {args.num_labels-1})")
    print(f"  Base vectors: {config['num_base']:,}")
    print(f"  Query vectors: {config['num_queries']:,}")
    print(f"  Groundtruth k: {args.k}")
    print(f"  Random seed: {args.seed}")
    print(f"\n[Paths]")
    print(f"  Base fvecs: {base_fvecs}")
    print(f"  Query fvecs: {query_fvecs}")
    print(f"  Output directory: {output_dir}")
    print()

    # Verify input files exist
    if not base_fvecs.exists():
        print(f"Error: Base file not found: {base_fvecs}")
        return
    if not query_fvecs.exists():
        print(f"Error: Query file not found: {query_fvecs}")
        return

    # Get actual file info
    actual_base_count, actual_dim = get_fvecs_info(str(base_fvecs))
    actual_query_count, _ = get_fvecs_info(str(query_fvecs))

    print(f"[File Validation]")
    print(f"  Actual base vectors: {actual_base_count:,}")
    print(f"  Actual query vectors: {actual_query_count:,}")
    print(f"  Actual dimension: {actual_dim}")

    base_size = min(config['num_base'], actual_base_count)
    query_size = min(config['num_queries'], actual_query_count)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths (Q-LEAF naming convention)
    base_labels_path = output_dir / "base_labels.txt"
    query_labels_path = output_dir / "query_labels.txt"
    groundtruth_path = output_dir / "groundtruth.ivecs"

    print("\n[Step 1] Generating labels...")
    print(f"  Generating {base_size:,} base labels...")
    base_labels = generate_labels(base_size, args.num_labels, seed=args.seed)
    print(f"  Generating {query_size:,} query labels...")
    query_labels = generate_labels(query_size, args.num_labels, seed=args.seed + 81)

    print("\nLabel distribution (base):")
    unique, counts = np.unique(base_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:,} ({count/base_size*100:.2f}%)")

    print("\nLabel distribution (query):")
    unique, counts = np.unique(query_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:,} ({count/query_size*100:.2f}%)")

    print("\n[Step 2] Saving labels...")
    save_labels(base_labels, str(base_labels_path))
    save_labels(query_labels, str(query_labels_path))

    if not args.skip_gt:
        print("\n[Step 3] Reading vectors...")
        base_vectors = read_fvecs_mmap(str(base_fvecs), max_count=base_size)
        query_vectors = read_fvecs_mmap(str(query_fvecs), max_count=query_size)

        print(f"  Base vectors shape: {base_vectors.shape}")
        print(f"  Query vectors shape: {query_vectors.shape}")

        print("\n[Step 4] Computing label-filtered groundtruth...")
        groundtruth = compute_label_groundtruth(
            base_vectors, query_vectors,
            base_labels, query_labels,
            k=args.k
        )

        print("\n[Step 5] Saving groundtruth...")
        save_ivecs(groundtruth, str(groundtruth_path))
    else:
        print("\n[Step 3-5] Skipping groundtruth computation (--skip-gt)")

    print("\n[Step 6] Creating symbolic links to original fvecs files...")
    import subprocess

    target_base_fvecs = output_dir / config["base_file"]
    target_query_fvecs = output_dir / config["query_file"]

    # Remove existing links if any
    subprocess.run(f"rm -f {target_base_fvecs}", shell=True)
    subprocess.run(f"rm -f {target_query_fvecs}", shell=True)

    # Create symbolic links
    subprocess.run(f"ln -s {base_fvecs} {target_base_fvecs}", shell=True)
    subprocess.run(f"ln -s {query_fvecs} {target_query_fvecs}", shell=True)

    print(f"  {target_base_fvecs} -> {base_fvecs}")
    print(f"  {target_query_fvecs} -> {query_fvecs}")

    print("\n" + "=" * 80)
    print("Dataset generation completed!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  Base labels:   {base_labels_path}")
    print(f"  Query labels:  {query_labels_path}")
    if not args.skip_gt:
        print(f"  Groundtruth:   {groundtruth_path}")
    print(f"  Base vectors:  {target_base_fvecs} (symlink)")
    print(f"  Query vectors: {target_query_fvecs} (symlink)")

    print(f"\n[File sizes]")
    print(f"  Base labels:  {os.path.getsize(base_labels_path):,} bytes")
    print(f"  Query labels: {os.path.getsize(query_labels_path):,} bytes")
    if not args.skip_gt:
        print(f"  Groundtruth:  {os.path.getsize(groundtruth_path):,} bytes")

    print(f"\n[Summary]")
    print(f"  Total labels: {args.num_labels}")
    print(f"  Vectors per label (avg): {base_size // args.num_labels:,}")
    print(f"  Output directory: {output_dir}")
    print()


if __name__ == "__main__":
    main()
