#!/usr/bin/env python3
"""
Multi-Dataset Adaptive NSG Subgraph Builder for Selectivity Experiment
"""

import argparse
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


# ============================================================
# DATASET_CONFIG
# ============================================================

@dataclass
class DatasetConfig:
    """Per-dataset configuration"""
    name: str                    
    dimension: int               
    num_base_vectors: int        
    num_queries: int             
    base_fvecs: Path             
    query_fvecs: Path            
    base_labels: Path            
    query_labels: Path           
    groundtruth: Path            
    output_dir: Path             


def get_dataset_configs() -> Dict[str, DatasetConfig]:
    """Return all dataset configurations"""

    # Base paths (using environment variables or defaults)
    dataset_base = get_dataset_dir()
    project_root = get_base_dir()
    label_base = project_root / "data"
    output_base = project_root / "NSG_subgraph_output"

    configs = {
        "sift10m": DatasetConfig(
            name="SIFT10M",
            dimension=128,
            num_base_vectors=10_000_000,
            num_queries=1_000,
            base_fvecs=dataset_base / "sift10m" / "sift10m_base.fvecs",
            query_fvecs=dataset_base / "sift10m" / "sift10m_query.fvecs",
            base_labels=label_base / "sift10m" / "base_labels_for_selectivity.txt",
            query_labels=label_base / "sift10m" / "query_labels_for_selectivity.txt",
            groundtruth=label_base / "sift10m" / "groundtruth_selectivity.ivecs",
            output_dir=output_base / "sift10m" / "adaptive_params"
        ),
        "deep10m": DatasetConfig(
            name="DEEP10M",
            dimension=96,
            num_base_vectors=9_990_000,
            num_queries=10_000,
            base_fvecs=dataset_base / "deep10m" / "deep-image-96-angular_base.fvecs",
            query_fvecs=dataset_base / "deep10m" / "deep-image-96-angular_query.fvecs",
            base_labels=label_base / "deep10m" / "base_labels_for_selectivity.txt",
            query_labels=label_base / "deep10m" / "query_labels_for_selectivity.txt",
            groundtruth=label_base / "deep10m" / "groundtruth_selectivity.ivecs",
            output_dir=output_base / "deep10m" / "adaptive_params"
        ),
        "text2image10m": DatasetConfig(
            name="TEXT2IMAGE10M",
            dimension=200,
            num_base_vectors=10_000_000,
            num_queries=100_000,
            base_fvecs=dataset_base / "text2image10m" / "base.10M.fvecs",
            query_fvecs=dataset_base / "text2image10m" / "query.public.100K.fvecs",
            base_labels=label_base / "text2image10m" / "base_labels_for_selectivity.txt",
            query_labels=label_base / "text2image10m" / "query_labels_for_selectivity.txt",
            groundtruth=label_base / "text2image10m" / "groundtruth_selectivity.ivecs",
            output_dir=output_base / "text2image10m" / "adaptive_params"
        ),
        "yfcc10m": DatasetConfig(
            name="YFCC10M",
            dimension=192,
            num_base_vectors=10_000_000,
            num_queries=100_000,
            base_fvecs=dataset_base / "yfcc10M" / "base.10M.fvecs",
            query_fvecs=dataset_base / "yfcc10M" / "query.public.100K.fvecs",
            base_labels=label_base / "yfcc10m" / "base_labels_for_selectivity.txt",
            query_labels=label_base / "yfcc10m" / "query_labels_for_selectivity.txt",
            groundtruth=label_base / "yfcc10m" / "groundtruth_selectivity.ivecs",
            output_dir=output_base / "yfcc10m" / "adaptive_params"
        ),
    }

    return configs


# Executable paths
QLEAF_DIR = get_base_dir()
BUILD_NSG_SUBGRAPHS = QLEAF_DIR / "build" / "bin" / "build_nsg_subgraphs"
COMPUTE_EP_NSG = QLEAF_DIR / "build" / "bin" / "compute_ep_nsg"

# Parameter presets by scale
SCALE_PRESETS = {
    "small": {
        # Selectivity ~0.5% (includes 0.01%, 0.1%), subgraph size ~50,000
        "min_vectors": 0,
        "max_vectors": 50000,
        "K": 20, "L": 30, "iter": 5, "S": 10, "R": 50,
        "L_nsg": 20, "R_nsg": 16, "C_nsg": 100,
        "num_hubs": 16,
        "description": "Small scale (<50K vectors, selectivity ~0.5%)"
    },
    "medium": {
        # Selectivity 0.5%~5% (includes 1%), subgraph size 50,000~500,000
        "min_vectors": 50000,
        "max_vectors": 500000,
        "K": 40, "L": 50, "iter": 8, "S": 10, "R": 100,
        "L_nsg": 30, "R_nsg": 32, "C_nsg": 100,
        "num_hubs": 32,
        "description": "Medium scale (50K-500K vectors, selectivity 0.5%~5%)"
    },
    "large": {
        # Selectivity 5% or more (includes 8%), subgraph size 500,000 or more
        "min_vectors": 500000,
        "max_vectors": float('inf'),
        "K": 150, "L": 150, "iter": 10, "S": 10, "R": 100,
        "L_nsg": 40, "R_nsg": 50, "C_nsg": 300,
        "num_hubs": 64,
        "description": "Large scale (>=500K vectors, selectivity >=5%)"
    }
}

# Scale processing order (large -> medium -> small)
SCALE_ORDER = ["large", "medium", "small"]


def get_scale_for_size(num_vectors: int) -> str:
    """Return scale name based on number of vectors"""
    if num_vectors >= 500000:
        return "large"
    elif num_vectors >= 50000:
        return "medium"
    else:
        return "small"


# Resource monitoring
def get_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB units (Linux)
    return usage.ru_maxrss / 1024.0


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h ({seconds:.0f}s)"


class Logger:
    def __init__(self, log_file: Path, dataset_name: str):
        self.log_file = log_file
        self.dataset_name = dataset_name
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        with open(self.log_file, 'w') as f:
            f.write(f"=== Multi-Dataset Adaptive NSG Build Log ===\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Started at: {datetime.now()}\n\n")

    def log(self, message: str, also_print: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        if also_print:
            print(full_message)

        with open(self.log_file, 'a') as f:
            f.write(full_message + "\n")

    def log_section(self, title: str):
        separator = "=" * 70
        self.log(separator)
        self.log(f"[{self.dataset_name}] {title}")
        self.log(separator)

    def log_resource_usage(self, task_name: str, elapsed_time: float):
        peak_mem = get_peak_memory_mb()
        self.log(f"\n{'─' * 50}")
        self.log(f"Resource Usage for {task_name} ({self.dataset_name}):")
        self.log(f"  Elapsed Time: {format_time(elapsed_time)}")
        self.log(f"  Peak Memory (RSS): {peak_mem:.2f} MB")
        self.log(f"{'─' * 50}\n")

    def log_final_summary(self):
        total_time = time.time() - self.start_time
        peak_mem = get_peak_memory_mb()

        self.log(f"\n{'═' * 70}")
        self.log(f"FINAL SUMMARY - {self.dataset_name}")
        self.log(f"{'═' * 70}")
        self.log(f"  Total Pipeline Time: {format_time(total_time)}")
        self.log(f"  Peak Memory (RSS): {peak_mem:.2f} MB")
        self.log(f"{'═' * 70}\n")


def analyze_labels(labels_file: Path, logger: Logger) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Read base_labels_for_selectivity.txt and determine scale group for each label
    """
    logger.log_section("Step 1: Analyzing Labels by Scale")
    label_counts = defaultdict(int)

    logger.log(f"Reading labels from: {labels_file}")
    with open(labels_file, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            logger.log(f"Detected header: num_vectors={parts[0]}, num_attributes={parts[1]}")
        else:
            logger.log(f"No header detected, treating first line as label data")
            if first_line:
                label_counts[first_line] += 1

        for line in f:
            label = line.strip()
            if label:
                label_counts[label] += 1

    total_labels = len(label_counts)
    total_vectors = sum(label_counts.values())
    logger.log(f"Total unique labels: {total_labels}")
    logger.log(f"Total vectors: {total_vectors:,}")

    # Group by scale
    scale_groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for label, count in label_counts.items():
        scale = get_scale_for_size(count)
        scale_groups[scale].append((label, count))

    for scale in scale_groups:
        try:
            scale_groups[scale].sort(key=lambda x: int(x[0]))
        except ValueError:
            scale_groups[scale].sort(key=lambda x: x[0])

    # Output statistics
    logger.log("\n" + "-" * 70)
    logger.log("Scale Group Distribution:")
    logger.log("-" * 70)
    logger.log(f"{'Scale':<10} {'Labels':<10} {'Vectors Range':<25} {'Total Vectors':<15}")
    logger.log("-" * 70)

    for scale_name in SCALE_ORDER:
        if scale_name in scale_groups:
            labels_in_group = scale_groups[scale_name]
            counts = [c for _, c in labels_in_group]
            min_count, max_count = min(counts), max(counts)
            total = sum(counts)
            logger.log(f"{scale_name:<10} {len(labels_in_group):<10} "
                      f"{min_count:,}~{max_count:,}".ljust(25) + f" {total:,}")

    logger.log("-" * 70)
    logger.log(f"{'TOTAL':<10} {total_labels:<10} {'':<25} {total_vectors:,}")
    logger.log("-" * 70)

    # Extract and return only label names
    result = {}
    for scale in scale_groups:
        result[scale] = [label for label, _ in scale_groups[scale]]

    return result, dict(label_counts)


def create_temp_labels_file(labels: List[str], base_labels_file: Path,
                            temp_dir: Path) -> Path:
    """
    Create temporary labels file for a specific label group
    Labels not belonging to the current scale group are replaced with "IGNORE"
    """
    temp_file = temp_dir / "temp_labels.txt"

    with open(base_labels_file, 'r') as f:
        original_lines = f.readlines()

    label_set = set(labels)

    start_idx = 0
    if original_lines:
        first_parts = original_lines[0].strip().split()
        if len(first_parts) == 2 and first_parts[0].isdigit() and first_parts[1].isdigit():
            start_idx = 1  # Skip header

    with open(temp_file, 'w') as f:
        for line in original_lines[start_idx:]:
            label = line.strip()
            if label in label_set:
                f.write(line)
            else:
                f.write("IGNORE\n")

    return temp_file


# Subgraph build
def run_command(cmd: List[str], logger: Logger, description: str) -> bool:
    """Execute command and capture logs"""
    logger.log(f"\n{'─' * 70}")
    logger.log(f"Running: {description}")
    logger.log(f"Command: {' '.join(str(c) for c in cmd)}")
    logger.log(f"{'─' * 70}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            logger.log(line.rstrip(), also_print=False)
            if any(keyword in line for keyword in ['Building', 'Completed', 'Error', 'Warning',
                                                     'Cluster', 'Memory', 'VmHWM', 'Step', '===',
                                                     'SUCCESS', 'FAILED', 'Scale:', 'vectors',
                                                     'Classified', 'label', 'Processing']):
                print(f"  {line.rstrip()}")

        process.wait()
        success = (process.returncode == 0)

    except Exception as e:
        logger.log(f"ERROR: {e}")
        success = False

    logger.log(f"{'─' * 70}")
    logger.log(f"Status: {'SUCCESS' if success else 'FAILED'}")
    logger.log(f"{'─' * 70}\n")

    return success


def build_subgraphs_for_scale(scale_name: str, labels: List[str],
                               dataset_config: DatasetConfig,
                               logger: Logger, builder_path: Path) -> bool:
    task_start = time.time()

    preset = SCALE_PRESETS[scale_name]
    output_dir = dataset_config.output_dir
    scale_output_dir = output_dir / scale_name

    logger.log_section(f"Task 1-{scale_name.upper()}: Building {len(labels)} Subgraphs")
    logger.log(f"Dataset: {dataset_config.name}")
    logger.log(f"Scale: {scale_name} ({preset['description']})")
    logger.log(f"Labels: {len(labels)} (range: {labels[0]} ~ {labels[-1]})")
    logger.log(f"Output: {scale_output_dir}")
    logger.log("")
    logger.log(f"NN-Descent: K={preset['K']}, L={preset['L']}, iter={preset['iter']}, "
               f"S={preset['S']}, R={preset['R']}")
    logger.log(f"NSG: R={preset['R_nsg']}, L={preset['L_nsg']}, C={preset['C_nsg']}")
    logger.log(f"Hub Nodes: {preset['num_hubs']}")

    # Check input files
    if not dataset_config.base_fvecs.exists():
        logger.log(f"ERROR: Base vectors file not found: {dataset_config.base_fvecs}")
        return False

    if not builder_path.exists():
        logger.log(f"ERROR: Builder executable not found: {builder_path}")
        return False

    # Create temporary labels file
    temp_dir = scale_output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_file = create_temp_labels_file(labels, dataset_config.base_labels, temp_dir)

    logger.log(f"Temp labels file: {temp_labels_file}")

    # Build command configuration
    dataset_name_with_scale = f"adaptive_params/{scale_name}"

    cmd = [
        str(builder_path),
        str(dataset_config.base_fvecs),
        str(temp_labels_file),
        str(output_dir.parent),  
        dataset_name_with_scale,  
        str(preset['K']),
        str(preset['L']),
        str(preset['iter']),
        str(preset['S']),
        str(preset['R']),
        str(preset['L_nsg']),
        str(preset['R_nsg']),
        str(preset['C_nsg']),
    ]

    success = run_command(cmd, logger, f"NSG Subgraph Build ({dataset_config.name}, {scale_name})")

    if temp_labels_file.exists():
        temp_labels_file.unlink()
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    if success:
        param_dir = (f"K{preset['K']}_L{preset['L']}_iter{preset['iter']}_"
                    f"S{preset['S']}_R{preset['R']}_NSG_L{preset['L_nsg']}_"
                    f"R{preset['R_nsg']}_C{preset['C_nsg']}")
        actual_output = output_dir.parent / dataset_name_with_scale / param_dir

        graph_files = list(actual_output.glob("*_nsg.graph"))
        logger.log(f"Created {len(graph_files)} graph files in {actual_output}")

        if len(graph_files) != len(labels):
            logger.log(f"WARNING: Expected {len(labels)}, got {len(graph_files)}")

    task_elapsed = time.time() - task_start
    logger.log_resource_usage(f"Task 1 Build ({scale_name})", task_elapsed)

    return success


# Entry Point generation
def generate_entry_points_for_scale(scale_name: str, dataset_config: DatasetConfig,
                                     logger: Logger, ep_generator_path: Path,
                                     num_hubs_override: Optional[int] = None) -> bool:
    task_start = time.time()

    preset = SCALE_PRESETS[scale_name]
    num_hubs = num_hubs_override if num_hubs_override is not None else preset['num_hubs']
    output_dir = dataset_config.output_dir

    param_dir = (f"K{preset['K']}_L{preset['L']}_iter{preset['iter']}_"
                f"S{preset['S']}_R{preset['R']}_NSG_L{preset['L_nsg']}_"
                f"R{preset['R_nsg']}_C{preset['C_nsg']}")

    graph_dir = output_dir / scale_name / param_dir
    ep_output_dir = graph_dir / f"entry_points_{num_hubs}"

    logger.log_section(f"Task 2-{scale_name.upper()}: Generating Entry Points")
    logger.log(f"Dataset: {dataset_config.name}")
    logger.log(f"Scale: {scale_name}")
    if num_hubs_override is not None:
        logger.log(f"Num clusters (hubs): {num_hubs} [override, preset={preset['num_hubs']}]")
    else:
        logger.log(f"Num clusters (hubs): {num_hubs} [preset default]")
    logger.log(f"Graph directory: {graph_dir}")
    logger.log(f"Output directory: {ep_output_dir}")

    if not graph_dir.exists():
        logger.log(f"ERROR: Graph directory not found: {graph_dir}")
        return False

    if not ep_generator_path.exists():
        logger.log(f"ERROR: EP generator not found: {ep_generator_path}")
        return False

    ep_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ep_generator_path),
        "--base-fvecs", str(dataset_config.base_fvecs),
        "--graph-dir", str(graph_dir),
        "--output-dir", str(ep_output_dir),
        "--num-clusters", str(num_hubs),
    ]

    success = run_command(cmd, logger, f"Entry Points Generation ({dataset_config.name}, {scale_name}, hubs={num_hubs})")

    if success:
        ep_files = list(ep_output_dir.glob("*_entry_points.txt"))
        logger.log(f"Created {len(ep_files)} entry point files")

    task_elapsed = time.time() - task_start
    logger.log_resource_usage(f"Task 2 EP Gen ({scale_name})", task_elapsed)

    return success


# Unified config.csv generation
def get_config_csv_path(output_dir: Path, num_hubs_override: Optional[int] = None) -> Path:
    """Return config csv file path"""
    if num_hubs_override is not None:
        return output_dir / f"config_ep{num_hubs_override}.csv"
    else:
        return output_dir / "config_adaptive.csv"


def generate_unified_config_csv(dataset_config: DatasetConfig, logger: Logger,
                                 num_hubs_override: Optional[int] = None) -> bool:
    task_start = time.time()
    output_dir = dataset_config.output_dir

    logger.log_section("Task 3: Generating Unified Config CSV")
    logger.log(f"Dataset: {dataset_config.name}")

    if num_hubs_override is not None:
        logger.log(f"num_hubs override: {num_hubs_override} (all scales)")
    else:
        logger.log("num_hubs: adaptive (per-scale preset defaults)")

    all_entries = []

    for scale_name in SCALE_ORDER:
        preset = SCALE_PRESETS[scale_name]
        num_hubs = num_hubs_override if num_hubs_override is not None else preset['num_hubs']

        param_dir = (f"K{preset['K']}_L{preset['L']}_iter{preset['iter']}_"
                    f"S{preset['S']}_R{preset['R']}_NSG_L{preset['L_nsg']}_"
                    f"R{preset['R_nsg']}_C{preset['C_nsg']}")

        graph_dir = output_dir / scale_name / param_dir
        ep_dir = graph_dir / f"entry_points_{num_hubs}"

        if not graph_dir.exists():
            logger.log(f"WARNING: Graph directory not found for {scale_name}: {graph_dir}")
            continue

        graph_files = sorted(graph_dir.glob("*_nsg.graph"))
        logger.log(f"{scale_name}: Found {len(graph_files)} graphs, EP dir: entry_points_{num_hubs}")

        for graph_file in graph_files:
            category = graph_file.stem.replace("_nsg", "")
            idx_file = graph_dir / f"{category}.idx"
            ep_file = ep_dir / f"{category}_entry_points.txt"

            if not idx_file.exists():
                logger.log(f"  WARNING: Missing idx file for {category}")
                continue

            if not ep_file.exists():
                logger.log(f"  WARNING: Missing entry points file for {category}")
                continue

            all_entries.append({
                'category': category,
                'graph_file': str(graph_file.resolve()),
                'idx_file': str(idx_file.resolve()),
                'entry_points_file': str(ep_file.resolve()),
                'scale': scale_name
            })

    if not all_entries:
        logger.log("ERROR: No valid entries found")
        return False

    try:
        all_entries.sort(key=lambda x: int(x['category']))
    except ValueError:
        all_entries.sort(key=lambda x: x['category'])

    config_csv_path = get_config_csv_path(output_dir, num_hubs_override)

    try:
        with open(config_csv_path, 'w') as f:
            f.write("#category,graph_file,idx_file,entry_points_file\n")

            for entry in all_entries:
                f.write(f"{entry['category']},{entry['graph_file']},"
                       f"{entry['idx_file']},{entry['entry_points_file']}\n")

        logger.log(f"\nConfig CSV generated successfully")
        logger.log(f"  Path: {config_csv_path.resolve()}")
        logger.log(f"  Total entries: {len(all_entries)}")

        # Statistics by scale
        scale_counts = defaultdict(int)
        for entry in all_entries:
            scale_counts[entry['scale']] += 1

        logger.log("\nEntries by scale:")
        for scale in SCALE_ORDER:
            if scale in scale_counts:
                logger.log(f"  {scale}: {scale_counts[scale]}")

        # Log resource usage
        task_elapsed = time.time() - task_start
        logger.log_resource_usage("Task 3 Config Gen", task_elapsed)

        return True

    except Exception as e:
        logger.log(f"ERROR: Failed to write config CSV: {e}")
        return False


def verify_all_outputs(dataset_config: DatasetConfig, expected_count: int,
                        logger: Logger, num_hubs_override: Optional[int] = None) -> bool:
    """Unified verification of all output files"""
    output_dir = dataset_config.output_dir

    logger.log_section("Verification: Checking All Output Files")
    logger.log(f"Dataset: {dataset_config.name}")

    total_graphs = 0
    total_idx = 0
    total_ep = 0

    for scale_name in SCALE_ORDER:
        preset = SCALE_PRESETS[scale_name]
        num_hubs = num_hubs_override if num_hubs_override is not None else preset['num_hubs']

        param_dir = (f"K{preset['K']}_L{preset['L']}_iter{preset['iter']}_"
                    f"S{preset['S']}_R{preset['R']}_NSG_L{preset['L_nsg']}_"
                    f"R{preset['R_nsg']}_C{preset['C_nsg']}")

        graph_dir = output_dir / scale_name / param_dir
        ep_dir = graph_dir / f"entry_points_{num_hubs}"

        if graph_dir.exists():
            graphs = list(graph_dir.glob("*_nsg.graph"))
            idxs = list(graph_dir.glob("*.idx"))
            eps = list(ep_dir.glob("*_entry_points.txt")) if ep_dir.exists() else []

            total_graphs += len(graphs)
            total_idx += len(idxs)
            total_ep += len(eps)

            logger.log(f"\n{scale_name.upper()} ({graph_dir}):")
            logger.log(f"  .graph files: {len(graphs)}")
            logger.log(f"  .idx files: {len(idxs)}")
            logger.log(f"  _entry_points.txt files: {len(eps)} (entry_points_{num_hubs})")

    # Check config file
    config_csv = get_config_csv_path(output_dir, num_hubs_override)
    config_entries = 0
    if config_csv.exists():
        with open(config_csv, 'r') as f:
            config_entries = sum(1 for line in f if not line.startswith('#'))

    logger.log(f"\n" + "=" * 50)
    logger.log(f"TOTAL COUNTS ({dataset_config.name}):")
    logger.log(f"  .graph files: {total_graphs} (expected: {expected_count})")
    logger.log(f"  .idx files: {total_idx} (expected: {expected_count})")
    logger.log(f"  _entry_points.txt files: {total_ep} (expected: {expected_count})")
    logger.log(f"  config entries: {config_entries} (expected: {expected_count})")
    logger.log(f"  config file: {config_csv.name} ({'EXISTS' if config_csv.exists() else 'MISSING'})")
    logger.log("=" * 50)

    # Verification result
    success = (total_graphs == expected_count and
               total_idx == expected_count and
               total_ep == expected_count and
               config_entries == expected_count and
               config_csv.exists())

    if success:
        logger.log(f"\nVERIFICATION PASSED: All {expected_count} subgraphs created successfully")
    else:
        logger.log(f"\nVERIFICATION FAILED: Missing or incorrect file counts")
        if total_graphs != expected_count:
            logger.log(f"  - Graph files: {total_graphs} != {expected_count}")
        if total_idx != expected_count:
            logger.log(f"  - Idx files: {total_idx} != {expected_count}")
        if total_ep != expected_count:
            logger.log(f"  - Entry point files: {total_ep} != {expected_count}")

    return success


def verify_subgraphs_exist_for_scale(scale_name: str, output_dir: Path, logger: Logger) -> bool:
    """Check if subgraphs exist for a specific scale group"""
    preset = SCALE_PRESETS[scale_name]
    param_dir = (f"K{preset['K']}_L{preset['L']}_iter{preset['iter']}_"
                f"S{preset['S']}_R{preset['R']}_NSG_L{preset['L_nsg']}_"
                f"R{preset['R_nsg']}_C{preset['C_nsg']}")
    graph_dir = output_dir / scale_name / param_dir

    if not graph_dir.exists():
        logger.log(f"ERROR: Graph directory not found for {scale_name}: {graph_dir}")
        return False

    graph_files = list(graph_dir.glob("*_nsg.graph"))
    idx_files = list(graph_dir.glob("*.idx"))

    if not graph_files or not idx_files:
        logger.log(f"ERROR: No graph/idx files found for {scale_name} in {graph_dir}")
        return False

    logger.log(f"  {scale_name}: {len(graph_files)} graphs, {len(idx_files)} idx files")
    return True


def verify_entry_points_exist_for_scale(scale_name: str, output_dir: Path,
                                          logger: Logger, num_hubs: int) -> bool:
    """Check if Entry Points exist for a specific scale group"""
    preset = SCALE_PRESETS[scale_name]
    param_dir = (f"K{preset['K']}_L{preset['L']}_iter{preset['iter']}_"
                f"S{preset['S']}_R{preset['R']}_NSG_L{preset['L_nsg']}_"
                f"R{preset['R_nsg']}_C{preset['C_nsg']}")
    ep_dir = output_dir / scale_name / param_dir / f"entry_points_{num_hubs}"

    if not ep_dir.exists():
        logger.log(f"ERROR: EP directory not found for {scale_name}: {ep_dir}")
        return False

    ep_files = list(ep_dir.glob("*_entry_points.txt"))
    if not ep_files:
        logger.log(f"ERROR: No entry point files found for {scale_name} in {ep_dir}")
        return False

    logger.log(f"  {scale_name}: {len(ep_files)} entry point files (entry_points_{num_hubs})")
    return True


def print_scale_presets(logger: Logger):
    """Output scale preset table"""
    logger.log("\nAdaptive Parameter Configuration:")
    logger.log("-" * 100)
    logger.log(f"{'Scale':<8} {'Vector Range':<20} {'NN-Descent (K,L,iter,S,R)':<35} {'NSG (R,L,C)':<20} {'Hubs':<6}")
    logger.log("-" * 100)

    for scale_name in SCALE_ORDER:
        preset = SCALE_PRESETS[scale_name]
        if preset['max_vectors'] == float('inf'):
            vec_range = f">={preset['min_vectors']:,}"
        else:
            vec_range = f"{preset['min_vectors']:,}~{preset['max_vectors']:,}"

        nn_params = f"K={preset['K']}, L={preset['L']}, iter={preset['iter']}, S={preset['S']}, R={preset['R']}"
        nsg_params = f"R={preset['R_nsg']}, L={preset['L_nsg']}, C={preset['C_nsg']}"

        logger.log(f"{scale_name:<8} {vec_range:<20} {nn_params:<35} {nsg_params:<20} {preset['num_hubs']:<6}")

    logger.log("-" * 100)


def print_dataset_info(dataset_config: DatasetConfig, logger: Logger):
    """Output dataset information"""
    logger.log("\nDataset Configuration:")
    logger.log("-" * 70)
    logger.log(f"  Name: {dataset_config.name}")
    logger.log(f"  Dimension: {dataset_config.dimension}")
    logger.log(f"  Base Vectors: {dataset_config.num_base_vectors:,}")
    logger.log(f"  Queries: {dataset_config.num_queries:,}")
    logger.log(f"  Base Fvecs: {dataset_config.base_fvecs}")
    logger.log(f"  Query Fvecs: {dataset_config.query_fvecs}")
    logger.log(f"  Base Labels: {dataset_config.base_labels}")
    logger.log(f"  Query Labels: {dataset_config.query_labels}")
    logger.log(f"  Ground Truth: {dataset_config.groundtruth}")
    logger.log(f"  Output Dir: {dataset_config.output_dir}")
    logger.log("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Adaptive NSG Subgraph Builder for Selectivity Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter
  )

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sift10m", "deep10m", "text2image10m", "yfcc10m"],
                        help="Dataset to process")

    parser.add_argument("--builder", type=Path, default=BUILD_NSG_SUBGRAPHS,
                        help="Path to build_nsg_subgraphs executable")
    parser.add_argument("--ep-generator", type=Path, default=COMPUTE_EP_NSG,
                        help="Path to compute_ep_nsg executable")

    parser.add_argument("--task", type=str,
                        choices=["build", "entry-points", "config", "all"],
                        default="all",
                        help="Task to execute (default: all)")

    # Entry Point count override
    parser.add_argument("--num-hubs", type=int, default=None,
                        help="Override num_hubs for all scales")

    parser.add_argument("--expected-labels", type=int, default=None,
                        help="Expected number of labels (auto-detected if not specified)")

    args = parser.parse_args()

    dataset_configs = get_dataset_configs()
    dataset_config = dataset_configs[args.dataset]

    dataset_config.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = dataset_config.output_dir / f"adaptive_build_log_{timestamp}.txt"
    logger = Logger(log_file, dataset_config.name)

    task = args.task
    num_hubs_override = args.num_hubs

    logger.log_section("Multi-Dataset Adaptive NSG Build Pipeline")
    print_dataset_info(dataset_config, logger)

    logger.log(f"Selected task: {task}")
    if num_hubs_override is not None:
        logger.log(f"num_hubs override: {num_hubs_override} (all scales)")
    else:
        logger.log(f"num_hubs: adaptive (per-scale preset defaults)")
    logger.log(f"Log file: {log_file}")

    print_scale_presets(logger)

    if not dataset_config.base_fvecs.exists():
        logger.log(f"ERROR: Base vectors file not found: {dataset_config.base_fvecs}")
        sys.exit(1)

    if not dataset_config.base_labels.exists():
        logger.log(f"ERROR: Base labels file not found: {dataset_config.base_labels}")
        sys.exit(1)

    # Classify labels
    scale_groups, label_counts = analyze_labels(dataset_config.base_labels, logger)

    # Determine expected label count
    expected_labels = args.expected_labels
    if expected_labels is None:
        expected_labels = len(label_counts)
        logger.log(f"Auto-detected expected labels: {expected_labels}")

    # Build subgraphs for each scale group
    if task == "build" or task == "all":
        for scale_name in SCALE_ORDER:
            if scale_name not in scale_groups:
                logger.log(f"\nNo labels for scale '{scale_name}', skipping...")
                continue

            labels = scale_groups[scale_name]
            if not build_subgraphs_for_scale(scale_name, labels, dataset_config,
                                              logger, args.builder):
                logger.log(f"\nPipeline FAILED at Task 1 ({scale_name})")
                sys.exit(1)

    # Generate Entry Points for each scale group
    if task == "entry-points" or task == "all":
        if task == "entry-points":
            logger.log("\nVerifying subgraphs exist...")
            for scale_name in SCALE_ORDER:
                if scale_name not in scale_groups:
                    continue
                if not verify_subgraphs_exist_for_scale(scale_name, dataset_config.output_dir, logger):
                    logger.log("\nCannot run entry-points task: subgraphs not found")
                    logger.log("  Please run with --task build first, or use --task all")
                    sys.exit(1)

        for scale_name in SCALE_ORDER:
            if scale_name not in scale_groups:
                continue

            if not generate_entry_points_for_scale(scale_name, dataset_config,
                                                    logger, args.ep_generator,
                                                    num_hubs_override):
                logger.log(f"\nPipeline FAILED at Task 2 ({scale_name})")
                sys.exit(1)

    # Generate unified config
    if task == "config" or task == "all":
        if task == "config":
            logger.log("\nVerifying subgraphs and entry points exist...")
            for scale_name in SCALE_ORDER:
                if scale_name not in scale_groups:
                    continue
                if not verify_subgraphs_exist_for_scale(scale_name, dataset_config.output_dir, logger):
                    logger.log("\nCannot run config task: subgraphs not found")
                    sys.exit(1)

                num_hubs = num_hubs_override if num_hubs_override is not None else SCALE_PRESETS[scale_name]['num_hubs']
                if not verify_entry_points_exist_for_scale(scale_name, dataset_config.output_dir, logger, num_hubs):
                    logger.log("\nCannot run config task: entry points not found")
                    logger.log("  Please run with --task entry-points first")
                    sys.exit(1)

        if not generate_unified_config_csv(dataset_config, logger, num_hubs_override):
            logger.log("\nPipeline FAILED at Task 3 (Config CSV Generation)")
            sys.exit(1)

    # Verification (only for all or config task)
    if task == "all" or task == "config":
        verify_all_outputs(dataset_config, expected_labels, logger, num_hubs_override)

    logger.log_final_summary()

    config_csv_path = get_config_csv_path(dataset_config.output_dir, num_hubs_override)

    logger.log_section("Pipeline Completed")
    logger.log(f"Dataset: {dataset_config.name}")
    logger.log(f"Output directory: {dataset_config.output_dir}")
    logger.log(f"Config file: {config_csv_path}")
    logger.log(f"Log file: {log_file}")

    if task in ("all", "config"):
        logger.log("\n" + "=" * 70)
        logger.log("Next Steps:")
        logger.log("=" * 70)
        logger.log(f"\nRun benchmark with nsg_qepo_single_ep:")
        logger.log(f"  cd {NSG_YHS_DIR}/nsg_qepo")
        logger.log(f"  ./nsg_qepo_single_ep \\")
        logger.log(f"    {dataset_config.base_fvecs} \\")
        logger.log(f"    {config_csv_path} \\")
        logger.log(f"    {dataset_config.query_fvecs} \\")
        logger.log(f"    {dataset_config.query_labels} \\")
        logger.log(f"    10 \"50,100,150,200\" \\")
        logger.log(f"    {dataset_config.groundtruth}")


if __name__ == "__main__":
    main()
