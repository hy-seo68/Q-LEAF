#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


# Scale-based parameter presets
SCALE_PRESETS = {
    "small": {
        # Selectivity ~0.01%, subgraph size ~1,000
        "K": 20, "L": 30, "iter": 5, "S": 10, "R": 50,
        "L_nsg": 20, "R_nsg": 16, "C_nsg": 100,
        "num_hubs": 16,
        "description": "Small scale (~1K vectors, selectivity ~0.01%)"
    },
    "medium": {
        # Selectivity 0.1%~1%, subgraph size ~10k~100k
        "K": 40, "L": 50, "iter": 5, "S": 10, "R": 100,
        "L_nsg": 30, "R_nsg": 32, "C_nsg": 100,
        "num_hubs": 32,
        "description": "Medium scale (~10K-100K vectors, selectivity 0.1%~1%)"
    },
    "large": {
        # Selectivity ~10%, subgraph size ~1,000,000
        "K": 150, "L": 150, "iter": 10, "S": 10, "R": 100,
        "L_nsg": 40, "R_nsg": 50, "C_nsg": 300,
        "num_hubs": 64,
        "description": "Large scale (~1M vectors, selectivity ~10%)"
    }
}


def get_scale_preset(scale: str) -> dict:
    """Return parameter preset by scale name"""
    if scale not in SCALE_PRESETS:
        raise ValueError(f"Unknown scale '{scale}'. Available: {list(SCALE_PRESETS.keys())}")
    return SCALE_PRESETS[scale]


@dataclass
class BuildConfig:
    """Build configuration"""
    # Paths
    base_fvecs: Path
    labels_file: Path
    output_dir: Path
    dataset_name: str

    # Number of labels (included in directory name: {dataset_name}_label{num_labels})
    num_labels: int = 100

    # Scale setting (None means manual parameter mode)
    scale: Optional[str] = None

    # NN-Descent parameters
    K: int = 150
    L: int = 150
    iter: int = 10
    S: int = 10
    R: int = 100

    # NSG parameters
    L_nsg: int = 40
    R_nsg: int = 50
    C_nsg: int = 300

    # Entry Point parameters
    num_hubs: int = 60

    @classmethod
    def from_scale(cls, scale: str, base_fvecs: Path, labels_file: Path,
                   output_dir: Path, dataset_name: str, num_labels: int = 100) -> 'BuildConfig':
        """Create BuildConfig from scale preset"""
        preset = get_scale_preset(scale)
        return cls(
            base_fvecs=base_fvecs,
            labels_file=labels_file,
            output_dir=output_dir,
            dataset_name=dataset_name,
            num_labels=num_labels,
            scale=scale,
            K=preset["K"],
            L=preset["L"],
            iter=preset["iter"],
            S=preset["S"],
            R=preset["R"],
            L_nsg=preset["L_nsg"],
            R_nsg=preset["R_nsg"],
            C_nsg=preset["C_nsg"],
            num_hubs=preset["num_hubs"],
        )

    @property
    def param_dir_name(self) -> str:
        return (f"K{self.K}_L{self.L}_iter{self.iter}_S{self.S}_R{self.R}"
                f"_NSG_L{self.L_nsg}_R{self.R_nsg}_C{self.C_nsg}")

    @property
    def dataset_dir_name(self) -> str:
        return f"{self.dataset_name}_label{self.num_labels}"

    @property
    def graph_output_dir(self) -> Path:
        return self.output_dir / self.dataset_dir_name / self.param_dir_name

    @property
    def ep_output_dir(self) -> Path:
        return self.graph_output_dir / f"entry_points_{self.num_hubs}"


class Logger:
    """Logging and time measurement"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, 'w') as f:
            f.write(f"=== NSG Build & EP Generation Log ===\n")
            f.write(f"Started at: {datetime.now()}\n\n")

    def log(self, message: str, also_print: bool = True):
        """Print log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        if also_print:
            print(full_message)

        with open(self.log_file, 'a') as f:
            f.write(full_message + "\n")

    def log_section(self, title: str):
        """Print section header"""
        separator = "=" * 70
        self.log(separator)
        self.log(title)
        self.log(separator)


def run_command(cmd: List[str], logger: Logger, description: str) -> bool:
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

        # Real-time output and log capture
        for line in process.stdout:
            logger.log(line.rstrip(), also_print=False)
            # Print only important lines to console
            if any(keyword in line for keyword in ['Building', 'Completed', 'Error', 'Warning',
                                                     'Cluster', 'Memory', 'VmHWM', 'Step', '===',
                                                     'SUCCESS', 'FAILED']):
                print(f"  {line.rstrip()}")

        process.wait()
        success = (process.returncode == 0)

    except Exception as e:
        logger.log(f"ERROR: {e}")
        success = False

    logger.log(f"{'─' * 70}")
    logger.log(f"Status: {'SUCCESS' if success else 'FAILED'}")
    logger.log(f"Note: Detailed timing and memory statistics are logged by C++ process above")
    logger.log(f"{'─' * 70}\n")

    return success


def build_subgraphs(config: BuildConfig, logger: Logger,
                     builder_path: Path) -> bool:

    logger.log_section("TASK 1: Building NSG Subgraphs")

    logger.log(f"Dataset: {config.dataset_name}")
    logger.log(f"Base vectors: {config.base_fvecs}")
    logger.log(f"Labels file: {config.labels_file}")
    logger.log(f"Output directory: {config.graph_output_dir}")
    logger.log("")
    logger.log(f"NN-Descent Parameters: K={config.K}, L={config.L}, iter={config.iter}, "
               f"S={config.S}, R={config.R}")
    logger.log(f"NSG Parameters: L={config.L_nsg}, R={config.R_nsg}, C={config.C_nsg}")

    # Check input file existence
    if not config.base_fvecs.exists():
        logger.log(f"ERROR: Base vectors file not found: {config.base_fvecs}")
        return False

    if not config.labels_file.exists():
        logger.log(f"ERROR: Labels file not found: {config.labels_file}")
        return False

    if not builder_path.exists():
        logger.log(f"ERROR: Builder executable not found: {builder_path}")
        return False

    cmd = [
        str(builder_path),
        str(config.base_fvecs),
        str(config.labels_file),
        str(config.output_dir),
        config.dataset_dir_name,
        str(config.K),
        str(config.L),
        str(config.iter),
        str(config.S),
        str(config.R),
        str(config.L_nsg),
        str(config.R_nsg),
        str(config.C_nsg),
    ]

    success = run_command(cmd, logger, "NSG Subgraph Build")

    if success:
        logger.log(f"✓ Task 1 completed successfully")
    else:
        logger.log(f"✗ Task 1 failed")

    return success


def generate_entry_points(config: BuildConfig, logger: Logger,
                           ep_generator_path: Path) -> bool:

    logger.log_section("TASK 2: Generating Entry Points (Batch Mode)")

    logger.log(f"Entry Point Parameters: num_clusters={config.num_hubs}")
    logger.log(f"Graph directory: {config.graph_output_dir}")
    logger.log(f"Output directory: {config.ep_output_dir}")
    logger.log(f"Processing mode: SEQUENTIAL (FAISS internal parallelization)")

    # Create Entry Point output directory
    config.ep_output_dir.mkdir(parents=True, exist_ok=True)

    if not ep_generator_path.exists():
        logger.log(f"ERROR: EP generator not found: {ep_generator_path}")
        return False

    # Batch mode execution (single call processes all subgraphs)
    cmd = [
        str(ep_generator_path),
        "--base-fvecs", str(config.base_fvecs),
        "--graph-dir", str(config.graph_output_dir),
        "--output-dir", str(config.ep_output_dir),
        "--num-clusters", str(config.num_hubs),
    ]

    success = run_command(cmd, logger, "Entry Points Batch Generation")

    if success:
        logger.log(f"\n✓ Task 2 completed successfully")
    else:
        logger.log(f"\n✗ Task 2 failed")

    return success


def generate_config_csv(config: BuildConfig, logger: Logger) -> bool:

    logger.log_section("TASK 3: Generating config.csv")

    graph_dir = config.graph_output_dir
    ep_dir = config.ep_output_dir

    # File matching: {category}_nsg.graph, {category}.idx, {category}_entry_points.txt
    graph_files = sorted(graph_dir.glob("*_nsg.graph"))

    if not graph_files:
        logger.log("ERROR: No graph files found")
        return False

    # Config output path (separate file per num_hubs)
    config_csv_path = graph_dir / f"config_ep{config.num_hubs}.csv"

    logger.log(f"Output file: {config_csv_path}")
    logger.log(f"Found {len(graph_files)} graph files")
    logger.log("")

    csv_entries = []
    missing_files = []

    for graph_file in graph_files:
        # Extract category name (e.g., "1_indoor_daytime_nsg.graph" -> "1_indoor_daytime")
        category = graph_file.stem.replace("_nsg", "")

        # Build corresponding file paths
        idx_file = graph_dir / f"{category}.idx"
        ep_file = ep_dir / f"{category}_entry_points.txt"

        # Check file existence
        if not idx_file.exists():
            missing_files.append(f"Missing idx file: {idx_file}")
            logger.log(f"WARNING: {idx_file} not found")
            continue

        if not ep_file.exists():
            missing_files.append(f"Missing entry points file: {ep_file}")
            logger.log(f"WARNING: {ep_file} not found")
            continue

        # Create CSV entry (using absolute paths)
        csv_entries.append({
            'category': category,
            'graph_file': str(graph_file.absolute()),
            'idx_file': str(idx_file.absolute()),
            'entry_points_file': str(ep_file.absolute())
        })

        logger.log(f"✓ {category}")

    if missing_files:
        logger.log("\nWarnings:")
        for msg in missing_files:
            logger.log(f"  {msg}")

    if not csv_entries:
        logger.log("\nERROR: No valid entries to write")
        return False

    # Write config.csv file
    try:
        with open(config_csv_path, 'w') as f:
            # CSV header (commented out so C++ parser skips it)
            f.write("#category,graph_file,idx_file,entry_points_file\n")

            # Write each entry
            for entry in csv_entries:
                f.write(f"{entry['category']},{entry['graph_file']},"
                       f"{entry['idx_file']},{entry['entry_points_file']}\n")

        logger.log(f"\n✓ config.csv generated successfully")
        logger.log(f"  Path: {config_csv_path}")
        logger.log(f"  Entries: {len(csv_entries)}")

        # Sample output
        if csv_entries:
            logger.log(f"\nSample entry:")
            logger.log(f"  Category: {csv_entries[0]['category']}")
            logger.log(f"  Graph: {Path(csv_entries[0]['graph_file']).name}")
            logger.log(f"  Idx: {Path(csv_entries[0]['idx_file']).name}")
            logger.log(f"  Entry Points: {Path(csv_entries[0]['entry_points_file']).name}")

        return True

    except Exception as e:
        logger.log(f"\nERROR: Failed to write config.csv: {e}")
        return False


def verify_subgraphs_exist(config: BuildConfig, logger: Logger) -> bool:
    """Verify subgraph files exist (validation before entry-points task)"""
    graph_dir = config.graph_output_dir

    if not graph_dir.exists():
        logger.log(f"ERROR: Graph directory not found: {graph_dir}")
        return False

    graph_files = list(graph_dir.glob("*_nsg.graph"))
    idx_files = list(graph_dir.glob("*.idx"))

    if not graph_files:
        logger.log(f"ERROR: No .graph files found in {graph_dir}")
        return False

    if not idx_files:
        logger.log(f"ERROR: No .idx files found in {graph_dir}")
        return False

    logger.log(f"✓ Subgraphs verified: {len(graph_files)} graphs, {len(idx_files)} idx files")
    return True


def verify_entry_points_exist(config: BuildConfig, logger: Logger) -> bool:
    """Verify entry point files exist (validation before config task)"""
    ep_dir = config.ep_output_dir

    if not ep_dir.exists():
        logger.log(f"ERROR: Entry points directory not found: {ep_dir}")
        return False

    ep_files = list(ep_dir.glob("*_entry_points.txt"))

    if not ep_files:
        logger.log(f"ERROR: No entry points files found in {ep_dir}")
        return False

    logger.log(f"✓ Entry points verified: {len(ep_files)} files in {ep_dir.name}")
    return True


def verify_outputs(config: BuildConfig, logger: Logger) -> bool:
    """Verify output files"""
    logger.log_section("Verification: Checking Output Files")

    graph_dir = config.graph_output_dir
    ep_dir = config.ep_output_dir

    # Check .graph files
    graph_files = list(graph_dir.glob("*_nsg.graph"))
    idx_files = list(graph_dir.glob("*.idx"))
    ep_files = list(ep_dir.glob("*_entry_points.txt"))
    config_csv = graph_dir / f"config_ep{config.num_hubs}.csv"

    logger.log(f"Graph directory: {graph_dir}")
    logger.log(f"  .graph files: {len(graph_files)}")
    logger.log(f"  .idx files: {len(idx_files)}")

    logger.log(f"\nEntry Points directory: {ep_dir}")
    logger.log(f"  entry_points.txt files: {len(ep_files)}")

    logger.log(f"\nConfig file:")
    logger.log(f"  config.csv: {'✓ Found' if config_csv.exists() else '✗ Missing'}")

    # Check sample file size
    if graph_files:
        sample = graph_files[0]
        size_mb = sample.stat().st_size / 1024 / 1024
        logger.log(f"\nSample .graph file: {sample.name} ({size_mb:.2f} MB)")

    if ep_files:
        sample = ep_files[0]
        with open(sample, 'r') as f:
            lines = f.readlines()
        logger.log(f"Sample entry points file: {sample.name} ({len(lines)} lines)")

    if config_csv.exists():
        with open(config_csv, 'r') as f:
            lines = f.readlines()
        logger.log(f"Config CSV: {len(lines) - 1} entries (excluding header)")

    success = (len(graph_files) > 0 and len(idx_files) > 0 and
               len(ep_files) > 0 and config_csv.exists())

    if success:
        logger.log("\n✓ Verification passed: All expected outputs found")
    else:
        logger.log("\n✗ Verification failed: Missing output files")

    return success


def print_scale_presets():
    """Print available scale presets"""
    print("\nAvailable Scale Presets:")
    print("-" * 90)
    print(f"{'Scale':<10} {'NN-Descent (K,L,iter,S,R)':<30} {'NSG (R,L,C)':<20} {'Hubs':<8} Description")
    print("-" * 90)
    for scale_name, preset in SCALE_PRESETS.items():
        nn_params = f"K={preset['K']}, L={preset['L']}, iter={preset['iter']}, S={preset['S']}, R={preset['R']}"
        nsg_params = f"R={preset['R_nsg']}, L={preset['L_nsg']}, C={preset['C_nsg']}"
        print(f"{scale_name:<10} {nn_params:<30} {nsg_params:<20} {preset['num_hubs']:<8} {preset['description']}")
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="NSG Subgraph Builder & Entry Point Generator (Batch Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--dataset", required=False, default=None,
                        help="Dataset name (e.g., SIFT10M)")
    parser.add_argument("--num-labels", type=int, required=False, default=100,
                        help="Number of labels (output dir: {dataset}_label{num_labels}, default: 100)")
    parser.add_argument("--base-fvecs", type=Path, required=False, default=None,
                        help="Path to base.fvecs file")
    parser.add_argument("--labels", type=Path, required=False, default=None,
                        help="Path to labels.txt file")
    parser.add_argument("--output-dir", type=Path, required=False, default=None,
                        help="Output directory for graphs and entry points")

    # Task selection options
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument("--task", type=str, choices=["build", "entry-points", "config", "all"],
                            default="all",
                            help="Task to execute: "
                                 "build (subgraph only), "
                                 "entry-points (EP generation only), "
                                 "config (config.csv only), "
                                 "all (default, full pipeline)")

    # Scale-based parameter settings
    scale_group = parser.add_argument_group("Scale-based Parameters (Recommended)")
    scale_group.add_argument("--scale", type=str, choices=["small", "medium", "large"],
                             help="Scale preset: small (~1K), medium (~10K-100K), large (~1M). "
                                  "When specified, all NN-Descent, NSG, and Hub parameters are auto-configured.")
    scale_group.add_argument("--list-scales", action="store_true",
                             help="Show available scale presets and exit")

    # NN-Descent parameters (for manual configuration)
    nn_group = parser.add_argument_group("NN-Descent Parameters")
    nn_group.add_argument("--K", type=int, default=150,
                          help="KNN neighbors (default: 150)")
    nn_group.add_argument("--L", type=int, default=150,
                          help="Local join candidates (default: 150)")
    nn_group.add_argument("--iter", type=int, default=10,
                          help="NN-Descent iterations (default: 10)")
    nn_group.add_argument("--S", type=int, default=10,
                          help="Sample rate (default: 10)")
    nn_group.add_argument("--R", type=int, default=100,
                          help="Reverse neighbors (default: 100)")

    # NSG parameters
    nsg_group = parser.add_argument_group("NSG Parameters")
    nsg_group.add_argument("--L-nsg", type=int, default=40,
                          help="Search path length (default: 40)")
    nsg_group.add_argument("--R-nsg", type=int, default=50,
                          help="Max neighbors per node (default: 50)")
    nsg_group.add_argument("--C-nsg", type=int, default=300,
                          help="Candidate pool size (default: 300)")

    # Entry Point parameters
    ep_group = parser.add_argument_group("Entry Point Parameters")
    ep_group.add_argument("--num-hubs", type=int, default=None,
                          help="Number of entry points (clusters) (default: from scale preset, or 60 in manual mode)")

    # Executable paths
    exec_group = parser.add_argument_group("Executable Paths")
    exec_group.add_argument("--builder", type=Path,
                            default=Path("./build_nsg_subgraphs"),
                            help="Path to build_nsg_subgraphs executable")
    exec_group.add_argument("--ep-generator", type=Path,
                            default=Path("./build/bin/compute_ep_nsg"),
                            help="Path to compute_ep_nsg executable (batch mode)")

    args = parser.parse_args()

    # Handle --list-scales option
    if args.list_scales:
        print_scale_presets()
        sys.exit(0)

    # Validate required arguments
    if not args.dataset or not args.base_fvecs or not args.labels or not args.output_dir:
        parser.error("--dataset, --base-fvecs, --labels, --output-dir are required")

    # Create configuration: scale-based or manual parameters
    if args.scale:
        # Scale-based automatic parameter configuration
        print(f"\n[Scale Mode] Using '{args.scale}' preset")
        preset = get_scale_preset(args.scale)
        print(f"  NN-Descent: K={preset['K']}, L={preset['L']}, iter={preset['iter']}, S={preset['S']}, R={preset['R']}")
        print(f"  NSG: R={preset['R_nsg']}, L={preset['L_nsg']}, C={preset['C_nsg']}")
        print(f"  Hub Nodes: {preset['num_hubs']}")
        print(f"  Description: {preset['description']}\n")

        config = BuildConfig.from_scale(
            scale=args.scale,
            base_fvecs=args.base_fvecs,
            labels_file=args.labels,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            num_labels=args.num_labels,
        )

        # Override scale preset value if --num-hubs is specified
        if args.num_hubs is not None:
            print(f"  [Override] num_hubs: {preset['num_hubs']} → {args.num_hubs}")
            config.num_hubs = args.num_hubs
    else:
        # Manual parameter configuration (legacy mode)
        print("\n[Manual Mode] Using custom parameters")
        config = BuildConfig(
            base_fvecs=args.base_fvecs,
            labels_file=args.labels,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            num_labels=args.num_labels,
            scale=None,
            K=args.K,
            L=args.L,
            iter=args.iter,
            S=args.S,
            R=args.R,
            L_nsg=args.L_nsg,
            R_nsg=args.R_nsg,
            C_nsg=args.C_nsg,
            num_hubs=args.num_hubs if args.num_hubs is not None else 60,
        )

    # Log file setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.graph_output_dir / f"build_log_{timestamp}.txt"
    logger = Logger(log_file)

    # Start header
    logger.log_section(f"NSG Build Pipeline: {config.dataset_name}")
    logger.log(f"Configuration:")
    logger.log(f"  Base vectors: {config.base_fvecs}")
    logger.log(f"  Labels file: {config.labels_file}")
    logger.log(f"  Output directory: {config.output_dir}")
    logger.log(f"  Dataset directory: {config.dataset_dir_name}")
    logger.log(f"  Parameter directory: {config.param_dir_name}")
    logger.log(f"  Num labels: {config.num_labels}")
    logger.log(f"  Log file: {log_file}")
    if config.scale:
        logger.log(f"  Scale preset: {config.scale} ({SCALE_PRESETS[config.scale]['description']})")
    logger.log(f"  NN-Descent: K={config.K}, L={config.L}, iter={config.iter}, S={config.S}, R={config.R}")
    logger.log(f"  NSG: R={config.R_nsg}, L={config.L_nsg}, C={config.C_nsg}")
    logger.log(f"  Hub Nodes: {config.num_hubs}")
    logger.log("")

    pipeline_start = time.time()
    task = args.task

    logger.log(f"Selected task: {task}")
    logger.log("")

    # Execute tasks
    if task == "build" or task == "all":
        # Task 1: Build subgraphs
        if not build_subgraphs(config, logger, args.builder):
            logger.log("\n✗ Pipeline FAILED at Task 1 (Subgraph Build)")
            sys.exit(1)

    if task == "entry-points" or task == "all":
        # Task 2: Generate Entry Points
        if task == "entry-points":
            if not verify_subgraphs_exist(config, logger):
                logger.log("\n✗ Cannot run entry-points task: subgraphs not found")
                logger.log("  Please run with --task build first, or use --task all")
                sys.exit(1)

        if not generate_entry_points(config, logger, args.ep_generator):
            logger.log("\n✗ Pipeline FAILED at Task 2 (Entry Point Generation)")
            sys.exit(1)

    if task == "config" or task == "all":
        # Task 3: Generate config.csv
        if task == "config":
            if not verify_subgraphs_exist(config, logger):
                logger.log("\n✗ Cannot run config task: subgraphs not found")
                sys.exit(1)
            if not verify_entry_points_exist(config, logger):
                logger.log("\n✗ Cannot run config task: entry points not found")
                logger.log(f"  Expected directory: {config.ep_output_dir}")
                logger.log("  Please run with --task entry-points first")
                sys.exit(1)

        if not generate_config_csv(config, logger):
            logger.log("\n✗ Pipeline FAILED at Task 3 (Config CSV Generation)")
            sys.exit(1)

    # Verification (only for 'all' or 'config' task)
    if task == "all" or task == "config":
        verify_outputs(config, logger)

    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start

    # Final summary
    logger.log_section("Pipeline Completed Successfully")
    logger.log(f"Dataset: {config.dataset_name}")
    logger.log(f"Total pipeline time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    logger.log(f"Output directory: {config.graph_output_dir}")
    logger.log(f"Entry points directory: {config.ep_output_dir}")
    logger.log(f"Config file: {config.graph_output_dir / f'config_ep{config.num_hubs}.csv'}")
    logger.log(f"Log file: {log_file}")
    logger.log("\n✓ All tasks completed successfully!")
    logger.log("\nNext step:")
    logger.log(f"  Run nsg_qleaf_single_ep with: --config {config.graph_output_dir / f'config_ep{config.num_hubs}.csv'}")


if __name__ == "__main__":
    main()
