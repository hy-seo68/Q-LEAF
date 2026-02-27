#!/usr/bin/env python3
"""
NSG QEPO Single Entry Point Benchmark Execution Script
"""

import argparse
import subprocess
import sys
import os
import yaml
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_base_dir() -> Path:
    env_path = os.environ.get("QLEAF_BASE_DIR")
    if env_path:
        return Path(env_path)
    return get_project_root()

def get_dataset_config_path() -> Path:
    env_path = os.environ.get("QLEAF_DATASET_CONFIG")
    if env_path:
        return Path(env_path)
    return get_project_root() / "config" / "dataset_config"

class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color

def print_header(message: str):
    print(f"{Colors.BLUE}{'=' * 80}{Colors.NC}")
    print(f"{Colors.BLUE}{message}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.NC}")

def print_info(message: str):
    print(f"{Colors.GREEN}ℹ{Colors.NC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}✘{Colors.NC} {message}")

def print_success(message: str):
    print(f"{Colors.GREEN}✔{Colors.NC} {message}")


class NSGQEPOBenchmark:
    SEPARATOR_LINE = "=" * 50

    def __init__(self, dataset: str, num_labels: int, graph_config: str, k: int, search_l_values: List[int],
                 num_eps: int):
        self.dataset_base = dataset  
        self.num_labels = num_labels
        self.dataset = f"{self.dataset_base}_label{self.num_labels}"
        self.graph_config = graph_config
        self.k = k
        self.search_l_values = search_l_values
        self.num_eps = num_eps

        self.base_dir = get_base_dir()
        self.benchmark_executable = self.base_dir / "build/bin/nsg_qepo_single_ep"

        self.dataset_config_path = get_dataset_config_path()
        self.nsg_subgraphs_base_dir = self.base_dir / "NSG_subgraph_output"

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.dataset_dir = None
        self.base_fvecs_file = None  # Global Pool: original base.fvecs
        self.query_vec_file = None
        self.query_label_file = None
        self.ground_truth_file = None
        self.selected_param_dir = None
        self.entry_points_dir = None
        self.results_dir = None
        self.config_file = None

    def load_dataset_config(self) -> Dict:
        """Load dataset configuration file"""
        dataset_lower = self.dataset.lower()
        config_file = self.dataset_config_path / f"{dataset_lower}.yaml"

        if not config_file.exists():
            print_error(f"Dataset configuration file not found: {config_file}")
            print_info("Available datasets:")
            for yaml_file in self.dataset_config_path.glob("*.yaml"):
                if yaml_file.stem != "dataset_info":
                    print(f"  - {yaml_file.stem.upper()}")
            sys.exit(1)

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        print_success(f"Dataset configuration loaded: {config_file}")
        return config

    def _find_nsg_subgraph_dir(self) -> Optional[Path]:
        if not self.nsg_subgraphs_base_dir.exists():
            return None

        target_name_lower = self.dataset.lower()

        for subdir in self.nsg_subgraphs_base_dir.iterdir():
            if subdir.is_dir() and subdir.name.lower() == target_name_lower:
                return subdir

        return None

    def setup_dataset_paths(self):
        """Set up paths for each dataset"""
        config = self.load_dataset_config()

        self.dataset_dir = Path(config['dataset_path'])

        # Global Pool: original base.fvecs file
        self.base_fvecs_file = self.dataset_dir / config['vector_files']['base']

        self.query_vec_file = self.dataset_dir / config['vector_files']['query']
        self.query_label_file = self.dataset_dir / config['label_files']['query']
        self.ground_truth_file = self.dataset_dir / config['ground_truth_file']

        # NSG subgraph path
        nsg_subgraphs_root = self._find_nsg_subgraph_dir()
        if nsg_subgraphs_root is None:
            print_error(f"NSG subgraph directory not found: {self.dataset}")
            print_info("Available directories:")
            if self.nsg_subgraphs_base_dir.exists():
                for subdir in sorted(self.nsg_subgraphs_base_dir.iterdir()):
                    if subdir.is_dir():
                        print(f"  - {subdir.name}")
            sys.exit(1)

        self.selected_param_dir = nsg_subgraphs_root / self.graph_config

        # Entry Points directory (within subgraph directory)
        self.entry_points_dir = self.selected_param_dir / f"entry_points_{self.num_eps}"

        dataset_upper = f"{self.dataset_base.upper()}_label{self.num_labels}"
        self.results_dir = self.base_dir / "results" / dataset_upper / self.graph_config / f"ep{self.num_eps}_single_ep_{self.timestamp}"

        print_info(f"Dataset directory: {Colors.CYAN}{self.dataset_dir}{Colors.NC}")
        print_info(f"Base vector file: {Colors.CYAN}{self.base_fvecs_file}{Colors.NC}")
        print_info(f"Subgraph directory: {Colors.CYAN}{self.selected_param_dir}{Colors.NC}")
        print_info(f"Results directory: {Colors.CYAN}{self.results_dir}{Colors.NC}")

    def check_build(self):
        """Check executable file"""
        if not self.benchmark_executable.exists():
            print_error(f"Executable not built: {self.benchmark_executable}")
            print_info("Build commands:")
            print(f"  cd {self.base_dir}")
            print("  mkdir -p build && cd build")
            print("  cmake .. && make")
            sys.exit(1)

        print_success(f"Executable verified: {self.benchmark_executable.name}")

    def check_target_directory(self):
        """Check target directory"""
        if not self.selected_param_dir.exists():
            print_error(f"Specified graph configuration directory not found: {self.selected_param_dir}")
            print("")
            print("Available graph configurations:")

            nsg_subgraphs_root = self.nsg_subgraphs_base_dir / self.dataset
            if nsg_subgraphs_root.exists():
                for config_dir in sorted(nsg_subgraphs_root.iterdir()):
                    if config_dir.is_dir():
                        print(f"  - {config_dir.name}")
            else:
                print(f"  (Dataset directory does not exist: {nsg_subgraphs_root})")
            sys.exit(1)

        print_success(f"Target directory: {Colors.CYAN}{self.graph_config}{Colors.NC}")

    def _check_file_exists(self, file_path: Path, file_type: str) -> bool:
        """Check file existence (common function)"""
        if not file_path.exists():
            print_error(f"{file_type} file not found: {file_path}")
            return False
        print_success(f"{file_type} file verified: {Colors.CYAN}{file_path}{Colors.NC}")
        return True

    def _check_glob_files(self, directory: Path, pattern: str, file_type: str) -> bool:
        """Check files in directory using glob pattern (common function)"""
        files = list(directory.glob(pattern))
        if not files:
            print_error(f"{file_type} files not found: {directory}/{pattern}")
            return False
        print_success(f"{file_type}: {Colors.CYAN}{len(files)} files{Colors.NC} - {Colors.CYAN}{directory}{Colors.NC}")
        return True

    def check_data_files(self):
        """Check data files"""
        print_info("Checking data files...")

        files_to_check = [
            (self.base_fvecs_file, "Base vectors"),
            (self.query_vec_file, "Query vectors"),
            (self.query_label_file, "Query labels"),
            (self.ground_truth_file, "Ground Truth")
        ]

        all_exist = all(self._check_file_exists(file_path, file_type)
                       for file_path, file_type in files_to_check)

        if not all_exist:
            print_error("Required data files missing")
            sys.exit(1)

    def check_nsg_subgraphs(self):
        """Check NSG subgraphs"""
        print_info("Checking NSG subgraphs...")

        graph_exists = self._check_glob_files(self.selected_param_dir, "*_nsg.graph", "NSG graphs")

        idx_exists = self._check_glob_files(self.selected_param_dir, "*.idx", "Index Mappings (.idx)")

        ep_exists = True
        if not self.entry_points_dir.exists():
            print_error(f"Entry Points directory not found: {self.entry_points_dir}")
            print_info(f"Available Entry Points directories:")
            for ep_dir in sorted(self.selected_param_dir.glob("entry_points_*")):
                if ep_dir.is_dir():
                    print(f"  - {ep_dir.name}")
            ep_exists = False
        else:
            ep_exists = self._check_glob_files(self.entry_points_dir, "*_entry_points.txt", "Entry Points")

        if not all([graph_exists, idx_exists, ep_exists]):
            print_error("Required subgraph files missing")
            sys.exit(1)

    def setup_config_file(self):
        """Check config file"""
        config_file = self.selected_param_dir / "config.csv"

        if not config_file.exists():
            print_error(f"config.csv file not found: {config_file}")
            print("")
            print_info("To generate config.csv, first run the following script:")
            print(f"  python build_and_generate_eps.py --dataset {self.dataset_base} --num-labels {self.num_labels} ...")
            print("")
            sys.exit(1)

        self.config_file = config_file

        # Simple file validity check
        with open(self.config_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if len(lines) == 0:
            print_error("config.csv file is empty")
            print_info("Re-run build_and_generate_eps.py to regenerate config.csv")
            sys.exit(1)

        print_success(f"Config file verified: {len(lines)} categories")
        print_info(f"File path: {Colors.CYAN}{self.config_file}{Colors.NC}")

    def _get_csv_fieldnames(self) -> List[str]:
        """Return list of field names for CSV file"""
        return [
            'search_L', 'k', 'num_entry_points', 'total_queries',
            f'recall@{self.k}', 'pure_algorithm_qps', 'wall_clock_qps', 'overhead_percentage',
            'avg_phase1_ms', 'avg_phase2_ms', 'avg_phase3_ms',
            'phase1_percentage', 'phase2_percentage', 'phase3_percentage',
            'ep_distance_calculations', 'search_distance_calculations',
            'avg_ep_distance_per_query', 'avg_search_distance_per_query'
        ]

    def _parse_json_result(self, json_file: Path) -> Optional[Dict]:
        """Parse JSON result file and return as dictionary"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'experiments' in data and len(data['experiments']) > 0:
                exp = data['experiments'][0]
                results = exp.get('results', {})
                phase_timing = exp.get('phase_timing', {})
                efficiency = exp.get('efficiency_breakdown', {})
                params = data.get('parameters', {})

                return {
                    'data': data,
                    'exp': exp,
                    'results': results,
                    'phase_timing': phase_timing,
                    'efficiency': efficiency,
                    'params': params
                }
            return None

        except Exception as e:
            print_warning(f"Failed to read JSON file ({json_file.name}): {e}")
            return None

    def run_benchmark(self):
        """Run benchmark"""
        print_header("Starting benchmark execution")
        print("")
        print_info(f"Search parameter k: {self.k}")
        print_info(f"search_L values: {','.join(map(str, self.search_l_values))} (total {len(self.search_l_values)})")
        print("")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Results directory created: {self.results_dir}")
        print("")

        log_file = self.results_dir / "benchmark_log.txt"

        with open(log_file, 'w') as f:
            f.write(self.SEPARATOR_LINE + "\n")
            f.write("NSG QEPO Single Entry Point Benchmark\n")
            f.write("Architecture: Global Pool + Mapping\n")
            f.write(self.SEPARATOR_LINE + "\n")
            f.write(f"Execution start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Graph config: {self.graph_config}\n")
            f.write(f"Search parameter k: {self.k}\n")
            f.write(f"search_L values: {','.join(map(str, self.search_l_values))}\n")
            f.write(self.SEPARATOR_LINE + "\n\n")

        success_count = 0
        fail_count = 0

        csv_results = []

        # Run benchmark for each search_L value
        for search_l in self.search_l_values:
            print("")
            print_info(f"Running: search_L = {search_l}")
            print(f"  - k: {self.k}")
            print(f"  - config_file: {self.config_file}")
            print("")

            with open(log_file, 'a') as f:
                f.write("\n")
                f.write(self.SEPARATOR_LINE + "\n")
                f.write(f"search_L = {search_l}\n")
                f.write(self.SEPARATOR_LINE + "\n")
                f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Search parameters: k={self.k}, search_L={search_l}\n")
                f.write(self.SEPARATOR_LINE + "\n\n")

            # Usage: nsg_qepo_single_ep <base_fvecs_file> <config_file> <query_vec_file> <query_label_file> <k> <search_L_values> <ground_truth_file>
            cmd = [
                str(self.benchmark_executable),
                str(self.base_fvecs_file),      
                str(self.config_file),          
                str(self.query_vec_file),       
                str(self.query_label_file),     
                str(self.k),                    
                str(search_l),                  
                str(self.ground_truth_file)     
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.results_dir),
                    capture_output=True,
                    text=True,
                    check=True
                )

                with open(log_file, 'a') as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n=== STDERR ===\n")
                        f.write(result.stderr)

                print_success(f"search_L={search_l} benchmark completed")
                success_count += 1

                json_files = sorted(self.results_dir.glob(f"nsg_qepo_single_ep_*.json"))
                if json_files:
                    latest_json = json_files[-1] 
                    parsed = self._parse_json_result(latest_json)

                    if parsed:
                        results = parsed['results']
                        phase_timing = parsed['phase_timing']
                        efficiency = parsed['efficiency']
                        params = parsed['params']

                        csv_row = {
                            'search_L': search_l,
                            'k': self.k,
                            'num_entry_points': params.get('num_entry_points', self.num_eps),
                            'total_queries': results.get('total_queries', 0),
                            'recall@{}'.format(self.k): results.get(f'recall@{self.k}', 0),
                            'pure_algorithm_qps': results.get('pure_algorithm_qps', 0),
                            'wall_clock_qps': results.get('wall_clock_qps', 0),
                            'overhead_percentage': results.get('overhead_percentage', 0),
                            'avg_phase1_ms': phase_timing.get('avg_phase1_ms', 0),
                            'avg_phase2_ms': phase_timing.get('avg_phase2_ms', 0),
                            'avg_phase3_ms': phase_timing.get('avg_phase3_ms', 0),
                            'phase1_percentage': phase_timing.get('phase1_percentage', 0),
                            'phase2_percentage': phase_timing.get('phase2_percentage', 0),
                            'phase3_percentage': phase_timing.get('phase3_percentage', 0),
                            'ep_distance_calculations': efficiency.get('ep_distance_calculations', 0),
                            'search_distance_calculations': efficiency.get('search_distance_calculations', 0),
                            'avg_ep_distance_per_query': efficiency.get('avg_ep_distance_per_query', 0),
                            'avg_search_distance_per_query': efficiency.get('avg_search_distance_per_query', 0)
                        }
                        csv_results.append(csv_row)

            except subprocess.CalledProcessError as e:
                print_error(f"search_L={search_l} benchmark failed")

                with open(log_file, 'a') as f:
                    f.write(f"\n=== ERROR ===\n")
                    f.write(f"Return code: {e.returncode}\n")
                    f.write(f"STDOUT:\n{e.stdout}\n")
                    f.write(f"STDERR:\n{e.stderr}\n")

                fail_count += 1

        with open(log_file, 'a') as f:
            f.write("\n")
            f.write(self.SEPARATOR_LINE + "\n")
            f.write("Benchmark completed\n")
            f.write(self.SEPARATOR_LINE + "\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Succeeded: {success_count}\n")
            f.write(f"Failed: {fail_count}\n")
            f.write(self.SEPARATOR_LINE + "\n")

        csv_file = None
        if csv_results:
            csv_file = self.save_results_to_csv(csv_results)

        self.print_results_summary(success_count, fail_count, log_file, csv_file)


    def save_results_to_csv(self, csv_results: List[Dict]) -> Optional[Path]:
        if not csv_results:
            return None

        csv_file = self.results_dir / f"benchmark_summary_{self.timestamp}.csv"

        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._get_csv_fieldnames())
                writer.writeheader()
                writer.writerows(csv_results)

            print_success(f"CSV file created: {csv_file.name}")
            return csv_file

        except Exception as e:
            print_error(f"Failed to save CSV file: {e}")
            return None

    def print_results_summary(self, success_count: int, fail_count: int, log_file: Path, csv_file: Optional[Path] = None):
        print("")
        print_header("Benchmark Results Summary")
        print("")

        print_info(f"Results directory: {Colors.CYAN}{self.results_dir}{Colors.NC}")
        print("")

        json_files = sorted(self.results_dir.glob("nsg_qepo_single_ep_*.json"))

        if not json_files:
            print_warning("JSON result files not found")
        else:
            print_success("Generated result files:")
            print("")

            for json_file in json_files:
                print(f"{Colors.MAGENTA}📄 {json_file.name}{Colors.NC}")

                parsed = self._parse_json_result(json_file)
                if parsed:
                    results = parsed['results']
                    params = parsed['params']
                    phase_timing = parsed['phase_timing']

                    print("Results:")
                    print(f"    - Pure Algorithm QPS: {results.get('pure_algorithm_qps', 'N/A'):.2f}")
                    print(f"    - Wall-clock QPS: {results.get('wall_clock_qps', 'N/A'):.2f}")

                    k_val = params.get('k', 10)
                    recall_key = f'recall@{k_val}'
                    recall_val = results.get(recall_key, 'N/A')
                    if isinstance(recall_val, (int, float)):
                        print(f"    - Recall@{k_val}: {recall_val:.4f}")
                    else:
                        print(f"    - Recall@{k_val}: {recall_val}")

                    if phase_timing:
                        print("Phase Timing:")
                        print(f"    - Phase 1: {phase_timing.get('avg_phase1_ms', 'N/A'):.3f} ms ({phase_timing.get('phase1_percentage', 'N/A'):.1f}%)")
                        print(f"    - Phase 2: {phase_timing.get('avg_phase2_ms', 'N/A'):.3f} ms ({phase_timing.get('phase2_percentage', 'N/A'):.1f}%)")
                        print(f"    - Phase 3: {phase_timing.get('avg_phase3_ms', 'N/A'):.3f} ms ({phase_timing.get('phase3_percentage', 'N/A'):.1f}%)")

                print("")

        print_info("All files list:")
        for file in sorted(self.results_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                size_str = f"{size / 1024:.1f}K" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f}M"
                print(f"  - {file.name} ({size_str})")

        print("")
        print_header("All tasks completed")
        print("")
        print_success(f"Succeeded: {success_count}")
        if fail_count > 0:
            print_error(f"Failed: {fail_count}")
        print("")
        print_info(f"Full log: {Colors.CYAN}{log_file}{Colors.NC}")
        if csv_file:
            print_info(f"CSV summary: {Colors.CYAN}{csv_file}{Colors.NC}")
        print("")


def main():
    parser = argparse.ArgumentParser(
        description="NSG QEPO Single Entry Point Benchmark Execution Script (Global Pool + Mapping)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., SIFT10M, DEEP10M, Text2Image10M)"
    )

    parser.add_argument(
        "--num-labels",
        type=int,
        required=True,
        help="Number of labels (e.g., 12, 100)"
    )

    parser.add_argument(
        "--graph-config",
        type=str,
        required=True,
        help="NSG subgraph configuration directory name (e.g., K40_L50_iter5_S10_R100_NSG_L30_R32_C100)"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors to search (default: 10)"
    )

    parser.add_argument(
        "--search-l",
        type=str,
        default="10,30,50,70,90",
        help="NSG search list size (comma-separated, default: 10,30,50,70,90)"
    )

    parser.add_argument(
        "--num-eps",
        type=int,
        default=32,
        help="Number of Entry Points (default: 32)"
    )

    args = parser.parse_args()

    # Parse search_l values
    try:
        search_l_values = [int(x.strip()) for x in args.search_l.split(',')]
    except ValueError:
        print_error("--search-l value is invalid. Must be comma-separated integers.")
        sys.exit(1)

    # Run benchmark
    dataset_full_name = f"{args.dataset.upper()}_label{args.num_labels}"
    print_header(f"NSG QEPO Single Entry Point Benchmark ({dataset_full_name})")
    print("")
    print_info(f"Architecture: {Colors.CYAN}Global Pool + Mapping{Colors.NC}")
    print_info(f"Dataset: {Colors.CYAN}{dataset_full_name}{Colors.NC}")
    print_info(f"Number of labels: {Colors.CYAN}{args.num_labels}{Colors.NC}")
    print_info(f"Graph config: {Colors.CYAN}{args.graph_config}{Colors.NC}")
    print("")

    benchmark = NSGQEPOBenchmark(
        dataset=args.dataset,
        num_labels=args.num_labels,
        graph_config=args.graph_config,
        k=args.k,
        search_l_values=search_l_values,
        num_eps=args.num_eps
    )

    # Set up dataset paths
    benchmark.setup_dataset_paths()

    # Pre-execution checks
    benchmark.check_build()
    benchmark.check_target_directory()
    benchmark.check_data_files()
    benchmark.check_nsg_subgraphs()

    # Check config file
    benchmark.setup_config_file()

    print("")

    # Run benchmark
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
