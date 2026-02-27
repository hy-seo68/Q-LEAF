#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <thread>
#include <atomic>
#include <dirent.h>
#include <algorithm>
#include <unordered_map>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

// EFANNA/NSG library integration

// Step 1: efanna_graph headers (for NN-Descent - IndexRandom, IndexGraph)
#include <efanna2e/index.h>
#include <efanna2e/parameters.h>
#include <efanna2e/util.h>
#include <efanna2e/neighbor.h>        // Contains LockNeighbor definition
#include <efanna2e/index_random.h>
#include <efanna2e/index_graph.h>

// Step 2: Manually define SimpleNeighbor type from nsg's neighbor.h
namespace efanna2e {
    struct SimpleNeighbor {
        unsigned id;
        float distance;

        SimpleNeighbor() = default;
        SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

        inline bool operator<(const SimpleNeighbor &other) const {
            return distance < other.distance;
        }
    };

    struct SimpleNeighbors {
        std::vector<SimpleNeighbor> pool;
    };
}

// Step 3: nsg library header (for NSG graph building - IndexNSG)
#include <efanna2e/index_nsg.h>

// Step 4: QEPO data loader (for Global Pool + Mapping architecture - MmapVectorReader, file I/O functions)
#include "nsg_qepo_data_loader.h"

/**
 * @brief Configuration parameters structure
 */
struct BuildParams {
    // NN-Descent parameters
    unsigned K = 100;
    unsigned L = 100;
    unsigned iter = 10;
    unsigned S = 10;
    unsigned R = 100;

    // NSG parameters
    unsigned L_nsg = 50;
    unsigned R_nsg = 50;
    unsigned C_nsg = 500;
};


/**
 * @brief Read VmHWM (Peak RSS) from /proc/self/status
 * @return Peak RSS in KB, or -1 if failed
 */
long get_vmhwm_kb() {
    std::ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) {
        return -1;
    }

    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmHWM:") {
            std::istringstream iss(line.substr(6));
            long vmhwm_kb;
            iss >> vmhwm_kb;
            return vmhwm_kb;
        }
    }
    return -1;
}

/**
 * @brief Save original index mapping information to binary file (uint32_t format).
 */
void save_original_indices(const char* filename, const std::vector<uint32_t>& original_indices) {
    save_label_indices(filename, original_indices);
    std::cout << "Saved global index mapping to " << filename << std::endl;
}

/**
 * @brief Read labels.txt file and classify vectors by label
 */
std::unordered_map<std::string, std::vector<uint32_t>>
classify_vectors_by_label(const char* labels_file, size_t expected_num_vectors) {
    std::ifstream in(labels_file);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open labels file: " + std::string(labels_file));
    }

    std::unordered_map<std::string, std::vector<uint32_t>> label_map;
    std::string line;
    uint32_t index = 0;
    size_t ignore_count = 0;  // IGNORE label count

    if (std::getline(in, line)) {
        std::istringstream iss(line);
        size_t num_vectors_in_header, num_labels_in_header;
        std::string remainder;

        // Try parsing metadata header in "N K" format
        if ((iss >> num_vectors_in_header >> num_labels_in_header) && !(iss >> remainder)) {
            std::cout << "Detected metadata header: num_vectors=" << num_vectors_in_header
                      << ", num_labels=" << num_labels_in_header << std::endl;

            // Validation: Check if header vector count matches actual count
            if (num_vectors_in_header != expected_num_vectors) {
                std::cerr << "Warning: Header specifies " << num_vectors_in_header
                          << " vectors, but base.fvecs has " << expected_num_vectors
                          << " vectors. Proceeding with actual data size." << std::endl;
            }
        } else {
            std::cerr << "Warning: First line is not in 'N K' metadata format: \"" << line << "\"" << std::endl;
            std::cerr << "         Treating first line as label data." << std::endl;

            std::string category = convert_label_to_category(line);
            if (category == "IGNORE") {
                ignore_count++;
                index++;
            } else {
                label_map[category].push_back(index);
                index++;
            }
        }
    }

    while (std::getline(in, line)) {
        if (!line.empty()) {
            std::string category = convert_label_to_category(line);
            if (category == "IGNORE") {
                ignore_count++;
                index++;
                continue;  // Skip IGNORE labels
            }
            label_map[category].push_back(index);
            index++;
        }
    }

    in.close();

    // Validation: Total classified vectors + IGNORE count = expected vector count
    size_t total_classified = 0;
    for (const auto& pair : label_map) {
        total_classified += pair.second.size();
    }

    if (total_classified + ignore_count != expected_num_vectors) {
        throw std::runtime_error(
            "Label count mismatch: classified " + std::to_string(total_classified) +
            " + ignored " + std::to_string(ignore_count) +
            " = " + std::to_string(total_classified + ignore_count) +
            " vectors, but expected " + std::to_string(expected_num_vectors));
    }

    std::cout << "Classified " << total_classified << " vectors into "
              << label_map.size() << " labels";
    if (ignore_count > 0) {
        std::cout << " (ignored " << ignore_count << " vectors with IGNORE label)";
    }
    std::cout << ":" << std::endl;

    // Print vector count per label
    std::vector<std::pair<std::string, size_t>> label_counts;
    for (const auto& pair : label_map) {
        label_counts.push_back({pair.first, pair.second.size()});
    }
    std::sort(label_counts.begin(), label_counts.end());

    for (const auto& lc : label_counts) {
        std::cout << "  " << lc.first << ": " << lc.second << " vectors" << std::endl;
    }

    return label_map;
}

/**
 * @brief Build and save K-NN graph using EFANNA NN-Descent
 */
void build_knn_graph_with_efanna(const float* data, size_t num_vectors, size_t dim,
                                  const BuildParams& params, const char* output_file,
                                  double& knn_time) {
    std::cout << "\n=== Building KNN Graph with EFANNA NN-Descent ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  K (neighbors): " << params.K << std::endl;
    std::cout << "  L (local join candidates): " << params.L << std::endl;
    std::cout << "  iter (iterations): " << params.iter << std::endl;
    std::cout << "  S (sample rate): " << params.S << std::endl;
    std::cout << "  R (reverse neighbors): " << params.R << std::endl;

    // Check OpenMP threading
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (num_threads == 1) {
        std::cout << "EFANNA will run in single-threaded mode (external parallelization enabled)" << std::endl;
    } else {
        std::cout << "EFANNA will use " << num_threads << " threads (set by OMP_NUM_THREADS)" << std::endl;
    }
#else
    std::cout << "EFANNA will use default threading (OpenMP not available)" << std::endl;
#endif

    auto knn_start = std::chrono::high_resolution_clock::now();

    efanna2e::IndexRandom init_index(dim, num_vectors);
    init_index.Build(num_vectors, data, efanna2e::Parameters());

    efanna2e::IndexGraph index(dim, num_vectors, efanna2e::L2, (efanna2e::Index*)(&init_index));

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", params.K);
    paras.Set<unsigned>("L", params.L);
    paras.Set<unsigned>("iter", params.iter);
    paras.Set<unsigned>("S", params.S);
    paras.Set<unsigned>("R", params.R);

    std::cout << "Building KNN graph with NN-Descent..." << std::endl;
    index.Build(num_vectors, data, paras);

    auto knn_end = std::chrono::high_resolution_clock::now();
    knn_time = std::chrono::duration<double>(knn_end - knn_start).count();

    std::cout << "KNN graph construction completed in " << knn_time << " seconds" << std::endl;

    std::cout << "Saving KNN graph to " << output_file << std::endl;
    index.Save(output_file);

    std::cout << "KNN graph saved successfully" << std::endl;
}

/**
 * @brief Build NSG graph using EFANNA2E.
 */
void build_nsg_graph_with_efanna2e(const float* data, size_t num_vectors, size_t dim,
                                    const char* knn_graph_file, const BuildParams& params,
                                    const char* output_file, double& nsg_time) {
    std::cout << "\n=== Building NSG Graph with EFANNA2E ===" << std::endl;
    std::cout << "L = " << params.L_nsg << ", R = " << params.R_nsg
              << ", C = " << params.C_nsg << std::endl;

    auto nsg_start = std::chrono::high_resolution_clock::now();

    efanna2e::IndexNSG index(dim, num_vectors, efanna2e::L2, nullptr);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", params.L_nsg);
    paras.Set<unsigned>("R", params.R_nsg);
    paras.Set<unsigned>("C", params.C_nsg);
    paras.Set<std::string>("nn_graph_path", knn_graph_file);

    std::cout << "Building NSG index..." << std::endl;
    index.Build(num_vectors, data, paras);

    auto nsg_end = std::chrono::high_resolution_clock::now();
    nsg_time = std::chrono::duration<double>(nsg_end - nsg_start).count();

    std::cout << "NSG graph construction completed in " << nsg_time << " seconds" << std::endl;

    std::cout << "Saving NSG graph to " << output_file << std::endl;
    index.Save(output_file);

    std::cout << "NSG graph saved successfully" << std::endl;
}

/**
 * @brief Save build results to a text file.
 */
void save_build_results(const char* filename, const std::string& dataset_name,
                       const std::string& label_name, size_t num_vectors, size_t dim,
                       const BuildParams& params, double extract_time, double knn_time, double nsg_time) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + std::string(filename));
    }

    out << "=== NSG Subgraph Build Results (Global Pool + Mapping) ===" << std::endl;
    out << "Dataset: " << dataset_name << " / " << label_name << std::endl;
    out << "Vectors: " << num_vectors << std::endl;
    out << "Dimension: " << dim << std::endl;
    out << std::endl;

    out << "Build Parameters:" << std::endl;
    out << "  NN-Descent K: " << params.K << std::endl;
    out << "  NN-Descent L: " << params.L << std::endl;
    out << "  NN-Descent iter: " << params.iter << std::endl;
    out << "  NN-Descent S: " << params.S << std::endl;
    out << "  NN-Descent R: " << params.R << std::endl;
    out << "  NSG L: " << params.L_nsg << std::endl;
    out << "  NSG R: " << params.R_nsg << std::endl;
    out << "  NSG C: " << params.C_nsg << std::endl;
    out << std::endl;

    out << "Build Time (Detailed):" << std::endl;
    out << "  1. Vector extraction (from mmap): " << std::fixed << std::setprecision(2)
        << extract_time << " seconds" << std::endl;
    out << "  2. KNN graph construction (EFANNA NN-Descent): " << knn_time << " seconds" << std::endl;
    out << "  3. NSG graph construction (EFANNA2E): " << nsg_time << " seconds" << std::endl;
    out << "  Total build time: " << (extract_time + knn_time + nsg_time) << " seconds" << std::endl;
    out << std::endl;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    out << "Build completed at: " << std::ctime(&time_t);

    out.close();
    std::cout << "Build results saved to " << filename << std::endl;
}

/**
 * @brief Single subgraph build task parameters
 */
struct SubgraphBuildTask {
    // Global Pool + Mapping
    std::string base_fvecs_file;
    std::string idx_file;

    std::string output_dir;
    std::string dataset_name;
    std::string label_name;
    BuildParams params;

    // Result storage
    bool success = false;
    double extract_time = 0.0;
    double knn_time = 0.0;
    double nsg_time = 0.0;
    std::string error_message;
};

/**
 * @brief Create directory recursively
 */
bool create_directory_recursive(const std::string& path) {
    if (path.empty()) {
        std::cerr << "Error: Empty path provided to create_directory_recursive" << std::endl;
        return false;
    }

    if (path == "/" || path == "\\") {
        return true;
    }

    std::string clean_path = path;
    while (!clean_path.empty() && (clean_path.back() == '/' || clean_path.back() == '\\')) {
        clean_path.pop_back();
    }

    if (clean_path.empty()) {
        return true;
    }

    struct stat info;
    if (stat(clean_path.c_str(), &info) == 0) {
        if (info.st_mode & S_IFDIR) {
            return true;
        } else {
            std::cerr << "Error: Path exists but is not a directory: " << clean_path << std::endl;
            return false;
        }
    }

    size_t last_slash = clean_path.find_last_of("/\\");
    if (last_slash != std::string::npos && last_slash > 0) {
        std::string parent = clean_path.substr(0, last_slash);
        if (!create_directory_recursive(parent)) {
            return false;
        }
    }

    // Create current directory
    if (mkdir(clean_path.c_str(), 0755) != 0) {
        if (errno != EEXIST) {
            std::cerr << "Error: Cannot create directory " << clean_path
                     << " (" << strerror(errno) << ")" << std::endl;
            return false;
        }
    }

    return true;
}

/**
 * @brief Scan directory for .fvecs files and generate label list.
 */
std::vector<std::string> scan_fvecs_files(const std::string& base_data_dir,
                                          const std::string& base_mapping_dir) {
    std::vector<std::string> labels;

    // Open directory
    DIR* dir = opendir(base_data_dir.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory: " << base_data_dir << std::endl;
        return labels;
    }

    std::cout << "Scanning for .fvecs files in " << base_data_dir << std::endl;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);

        // Filter files with .fvecs extension only
        if (filename.size() > 6 && filename.substr(filename.size() - 6) == ".fvecs") {
            std::string label = filename.substr(0, filename.size() - 6);

            std::string fvecs_path = base_data_dir + "/" + filename;
            struct stat fvecs_stat;
            if (stat(fvecs_path.c_str(), &fvecs_stat) != 0 || !S_ISREG(fvecs_stat.st_mode)) {
                std::cerr << "Warning: Skipping " << filename << " (not a regular file)" << std::endl;
                continue;
            }

            // Verify corresponding .mapping file exists
            std::string mapping_path = base_mapping_dir + "/" + label + ".mapping";
            struct stat mapping_stat;
            if (stat(mapping_path.c_str(), &mapping_stat) != 0) {
                std::cerr << "Warning: Skipping " << label << " (no corresponding .mapping file: "
                         << mapping_path << ")" << std::endl;
                continue;
            }

            labels.push_back(label);
        }
    }

    closedir(dir);

    // Sort alphabetically (for reproducibility)
    std::sort(labels.begin(), labels.end());

    if (labels.empty()) {
        std::cerr << "Warning: No valid .fvecs files found in " << base_data_dir << std::endl;
        std::cerr << "Expected format: <label_name>.fvecs with corresponding <label_name>.mapping" << std::endl;
    } else {
        std::cout << "Found " << labels.size() << " valid labels:" << std::endl;
        for (const auto& label : labels) {
            std::cout << "  - " << label << std::endl;
        }
    }

    return labels;
}

/**
 * @brief Build a single subgraph
 */
void build_single_subgraph(SubgraphBuildTask& task) {
    try {
        // Disable EFANNA/EFANNA2E internal parallelization
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        setenv("OPENBLAS_NUM_THREADS", "1", 1);
        setenv("MKL_NUM_THREADS", "1", 1);

        std::cout << "\n[Thread " << std::this_thread::get_id() << "] "
                  << "Building subgraph: " << task.label_name << std::endl;

        // 1. Initialize global base.fvecs mmap (thread-safe, runs only once)
        static std::shared_ptr<MmapVectorReader> global_base_data;
        static std::mutex global_data_mutex;

        {
            std::lock_guard<std::mutex> lock(global_data_mutex);
            if (!global_base_data) {
                std::cout << "[Global mmap] Loading base.fvecs: " << task.base_fvecs_file << std::endl;
                auto load_start = std::chrono::high_resolution_clock::now();

                global_base_data = std::make_shared<MmapVectorReader>(task.base_fvecs_file.c_str());

                auto load_end = std::chrono::high_resolution_clock::now();
                double load_time = std::chrono::duration<double>(load_end - load_start).count();
                std::cout << "[Global mmap] Loaded " << global_base_data->num_vectors() << " vectors, "
                          << "dim=" << global_base_data->dimension()
                          << " (" << load_time << "s)" << std::endl;
            }
        }

        // 2. Load .idx file (global ID list per label)
        std::vector<uint32_t> label_indices = load_label_indices(task.idx_file.c_str());
        size_t num_vectors = label_indices.size();
        size_t dim = global_base_data->dimension();

        std::cout << "[Thread " << std::this_thread::get_id() << "] "
                  << task.label_name << ": " << num_vectors << " vectors" << std::endl;

        // Parameter validation
        if (num_vectors <= task.params.K) {
            throw std::runtime_error("K (" + std::to_string(task.params.K) +
                                   ") must be less than number of vectors (" +
                                   std::to_string(num_vectors) + ")");
        }

        // 3. Extract vectors (extract only vectors for this label from mmap to temporary array)
        std::cout << "[Thread " << std::this_thread::get_id() << "] "
                  << "Extracting " << num_vectors << " vectors from mmap..." << std::endl;

        auto extract_start = std::chrono::high_resolution_clock::now();

        // Allocate 32-byte aligned vector (AVX2 SIMD optimization)
        AlignedVector32<float> data(num_vectors * dim);
        for (size_t i = 0; i < num_vectors; i++) {
            const float* vec = global_base_data->get_vector(label_indices[i]);
            std::memcpy(data.data() + i * dim, vec, dim * sizeof(float));

            if (num_vectors > 100000 && ((i + 1) % 50000 == 0 || i == num_vectors - 1)) {
                std::cout << "  Extracted: " << (i + 1) << " / " << num_vectors
                         << " (" << ((i + 1) * 100 / num_vectors) << "%)" << std::endl;
            }
        }

        auto extract_end = std::chrono::high_resolution_clock::now();
        task.extract_time = std::chrono::duration<double>(extract_end - extract_start).count();
        std::cout << "[Thread " << std::this_thread::get_id() << "] "
                  << "Vector extraction completed (" << task.extract_time << "s)" << std::endl;

        // 4. Generate output file paths
        std::ostringstream param_dir_name;
        param_dir_name << "K" << task.params.K
                      << "_L" << task.params.L
                      << "_iter" << task.params.iter
                      << "_S" << task.params.S
                      << "_R" << task.params.R
                      << "_NSG_L" << task.params.L_nsg
                      << "_R" << task.params.R_nsg
                      << "_C" << task.params.C_nsg;

        std::string dataset_dir = task.output_dir + "/" + task.dataset_name + "/" + param_dir_name.str();
        std::string output_prefix = dataset_dir + "/" + task.label_name;

        if (!create_directory_recursive(dataset_dir)) {
            throw std::runtime_error("Failed to create output directory: " + dataset_dir);
        }

        std::string knn_graph_file = output_prefix + "_nsg.knng";
        std::string nsg_graph_file = output_prefix + "_nsg.graph";
        std::string indices_file = output_prefix + "_nsg.indices";
        std::string results_file = output_prefix + "_nsg_build.txt";

        // 5. Build KNN graph with EFANNA NN-Descent (single thread)
        build_knn_graph_with_efanna(data.data(), num_vectors, dim, task.params,
                                    knn_graph_file.c_str(), task.knn_time);

        // 6. Build NSG graph with EFANNA2E (single thread)
        build_nsg_graph_with_efanna2e(data.data(), num_vectors, dim,
                                      knn_graph_file.c_str(), task.params,
                                      nsg_graph_file.c_str(), task.nsg_time);

        save_original_indices(indices_file.c_str(), label_indices);

        save_build_results(results_file.c_str(), task.dataset_name, task.label_name,
                          num_vectors, dim, task.params, task.extract_time, task.knn_time, task.nsg_time);

        task.success = true;

        std::cout << "[Thread " << std::this_thread::get_id() << "] "
                  << "Completed: " << task.label_name
                  << " (KNN: " << task.knn_time << "s, NSG: " << task.nsg_time << "s)"
                  << std::endl;

    } catch (const std::exception& e) {
        task.success = false;
        task.error_message = e.what();
        std::cerr << "[Thread " << std::this_thread::get_id() << "] "
                  << "Failed: " << task.label_name << " - " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 13) {
        std::cerr << "========================================" << std::endl;
        std::cerr << "NSG Subgraph Builder (Global Pool + Mapping)" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <base_fvecs_file> <labels_file> <output_dir> "
                  << "<dataset_name> <K> <L> <iter> <S> <R> <L_nsg> <R_nsg> <C_nsg>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  base_fvecs_file : Original base.fvecs file (all vectors)" << std::endl;
        std::cerr << "  labels_file     : labels.txt file (one label per line)" << std::endl;
        std::cerr << "  output_dir      : Output directory for NSG graphs" << std::endl;
        std::cerr << "  dataset_name    : Dataset name (e.g., SIFT10M, GIST)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  NN-Descent Parameters:" << std::endl;
        std::cerr << "    K    : Number of neighbors for KNN graph (e.g., 100)" << std::endl;
        std::cerr << "    L    : Local join candidate size (e.g., 100)" << std::endl;
        std::cerr << "    iter : Number of NN-Descent iterations (e.g., 10)" << std::endl;
        std::cerr << "    S    : Sample rate (e.g., 10)" << std::endl;
        std::cerr << "    R    : Reverse neighbor size (e.g., 100)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  NSG Parameters:" << std::endl;
        std::cerr << "    L_nsg : Search path length for NSG construction (e.g., 50)" << std::endl;
        std::cerr << "    R_nsg : Max neighbors per node in NSG (e.g., 40)" << std::endl;
        std::cerr << "    C_nsg : Candidate pool size for NSG (e.g., 500)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " \\" << std::endl;
        std::cerr << "    data/sift10m/sift10m_base.fvecs \\" << std::endl;
        std::cerr << "    data/sift10m/base_labels.txt \\" << std::endl;        
        std::cerr << "    nsg_subgraphs_output SIFT10M \\" << std::endl;
        std::cerr << "    100 100 10 10 100 \\" << std::endl;
        std::cerr << "    50 40 500" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Output structure:" << std::endl;
        std::cerr << "  output_dir/SIFT10M/K100_L100_iter10_S10_R100_NSG_L50_R40_C500/" << std::endl;
        std::cerr << "    1_indoor_daytime.idx          (mapping file)" << std::endl;
        std::cerr << "    1_indoor_daytime_nsg.graph    (NSG graph)" << std::endl;
        std::cerr << "    1_indoor_daytime_nsg.knng     (KNN graph)" << std::endl;
        std::cerr << "    1_indoor_daytime_nsg.indices  (legacy mapping)" << std::endl;
        std::cerr << "========================================" << std::endl;
        return -1;
    }

    try {
        // Parse arguments
        std::string base_fvecs_file = argv[1];
        std::string labels_file = argv[2];
        std::string output_dir = argv[3];
        std::string dataset_name = argv[4];

        BuildParams params;

        // Parse NN-Descent parameters
        int k_val = atoi(argv[5]);
        int l_val = atoi(argv[6]);
        int iter_val = atoi(argv[7]);
        int s_val = atoi(argv[8]);
        int r_val = atoi(argv[9]);

        // Parse NSG parameters
        int l_nsg_val = atoi(argv[10]);
        int r_nsg_val = atoi(argv[11]);
        int c_nsg_val = atoi(argv[12]);

        if (k_val <= 0 || l_val <= 0 || iter_val <= 0 || s_val <= 0 || r_val <= 0 ||
            l_nsg_val <= 0 || r_nsg_val <= 0 || c_nsg_val <= 0) {
            throw std::runtime_error("All parameters must be positive integers");
        }

        params.K = static_cast<unsigned>(k_val);
        params.L = static_cast<unsigned>(l_val);
        params.iter = static_cast<unsigned>(iter_val);
        params.S = static_cast<unsigned>(s_val);
        params.R = static_cast<unsigned>(r_val);
        params.L_nsg = static_cast<unsigned>(l_nsg_val);
        params.R_nsg = static_cast<unsigned>(r_nsg_val);
        params.C_nsg = static_cast<unsigned>(c_nsg_val);

        std::cout << "\n========================================" << std::endl;
        std::cout << "NSG Subgraph Builder (Global Pool + Mapping)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Dataset: " << dataset_name << std::endl;
        std::cout << "NN-Descent Parameters: K=" << params.K << ", L=" << params.L
                  << ", iter=" << params.iter << ", S=" << params.S << ", R=" << params.R << std::endl;
        std::cout << "NSG Parameters: L=" << params.L_nsg << ", R=" << params.R_nsg
                  << ", C=" << params.C_nsg << std::endl;
        std::cout << "Base vectors: " << base_fvecs_file << std::endl;
        std::cout << "Labels file: " << labels_file << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Initial memory measurement
        long vmhwm_start_kb = get_vmhwm_kb();

        // STEP 1: Load base.fvecs to check vector count
        std::cout << "=== Step 1: Loading Base Vectors ===" << std::endl;
        auto base_load_start = std::chrono::high_resolution_clock::now();

        MmapVectorReader base_reader(base_fvecs_file.c_str());
        size_t total_vectors = base_reader.num_vectors();
        size_t dim = base_reader.dimension();

        auto base_load_end = std::chrono::high_resolution_clock::now();
        double base_load_time = std::chrono::duration<double>(base_load_end - base_load_start).count();

        std::cout << "Loaded " << total_vectors << " vectors, dim=" << dim
                  << " (" << base_load_time << "s)" << std::endl;
        std::cout << std::endl;

        // STEP 2: Read labels.txt and classify by label
        std::cout << "=== Step 2: Classifying Vectors by Label ===" << std::endl;
        auto classify_start = std::chrono::high_resolution_clock::now();

        auto label_map = classify_vectors_by_label(labels_file.c_str(), total_vectors);

        auto classify_end = std::chrono::high_resolution_clock::now();
        double classify_time = std::chrono::duration<double>(classify_end - classify_start).count();

        std::cout << "Classification completed in " << classify_time << " seconds" << std::endl;
        std::cout << std::endl;

        // STEP 3: Create output directory structure
        std::ostringstream param_dir_name;
        param_dir_name << "K" << params.K << "_L" << params.L << "_iter" << params.iter
                      << "_S" << params.S << "_R" << params.R
                      << "_NSG_L" << params.L_nsg << "_R" << params.R_nsg << "_C" << params.C_nsg;

        std::string dataset_dir = output_dir + "/" + dataset_name + "/" + param_dir_name.str();

        if (!create_directory_recursive(dataset_dir)) {
            throw std::runtime_error("Failed to create output directory: " + dataset_dir);
        }

        // STEP 4: Generate .idx files
        std::cout << "=== Step 3: Generating .idx Files ===" << std::endl;
        auto idx_gen_start = std::chrono::high_resolution_clock::now();

        std::vector<std::string> labels;
        for (const auto& pair : label_map) {
            std::string idx_file = dataset_dir + "/" + pair.first + ".idx";
            save_label_indices(idx_file.c_str(), pair.second);
            labels.push_back(pair.first);
        }

        auto idx_gen_end = std::chrono::high_resolution_clock::now();
        double idx_gen_time = std::chrono::duration<double>(idx_gen_end - idx_gen_start).count();

        std::cout << "Generated " << labels.size() << " .idx files in " << idx_gen_time << " seconds" << std::endl;
        std::cout << std::endl;

        // STEP 5: Create build tasks for each label
        std::sort(labels.begin(), labels.end());  // Sort for reproducibility

        std::vector<SubgraphBuildTask> tasks(labels.size());
        for (size_t i = 0; i < labels.size(); i++) {
            tasks[i].base_fvecs_file = base_fvecs_file;
            tasks[i].idx_file = dataset_dir + "/" + labels[i] + ".idx";
            // tasks[i].mapping_file = dataset_dir + "/" + labels[i] + "_nsg.indices";  // Saved later
            tasks[i].output_dir = output_dir;
            tasks[i].dataset_name = dataset_name;
            tasks[i].label_name = labels[i];
            tasks[i].params = params;
        }

        // STEP 6: Start parallel build (maximum 8 threads)
        const size_t MAX_THREADS = 8;
        const size_t num_threads = std::min(MAX_THREADS, tasks.size());

        auto build_start = std::chrono::high_resolution_clock::now();

        std::cout << "=== Step 4: Building NSG Subgraphs ===" << std::endl;
        std::cout << "Parallel build with " << num_threads << " threads" << std::endl;
        std::cout << "Total subgraphs to build: " << tasks.size() << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Track task index (atomic operation)
        std::atomic<size_t> next_task_idx(0);

        // Worker thread function
        auto worker = [&tasks, &next_task_idx]() {
            while (true) {
                size_t task_idx = next_task_idx.fetch_add(1);
                if (task_idx >= tasks.size()) {
                    break;
                }
                build_single_subgraph(tasks[task_idx]);
            }
        };

        // Create and run thread pool
        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back(worker);
        }

        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }

        auto build_end = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(build_end - build_start).count();

        // Results summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "Build Summary" << std::endl;
        std::cout << "========================================" << std::endl;

        int success_count = 0;
        int failure_count = 0;
        double total_extract_time = 0.0;
        double total_knn_time = 0.0;
        double total_nsg_time = 0.0;

        for (const auto& task : tasks) {
            if (task.success) {
                success_count++;
                total_extract_time += task.extract_time;
                total_knn_time += task.knn_time;
                total_nsg_time += task.nsg_time;
                std::cout << "✓ " << task.label_name
                          << " (Extract: " << std::fixed << std::setprecision(2)
                          << task.extract_time << "s, KNN: " << task.knn_time
                          << "s, NSG: " << task.nsg_time << "s)" << std::endl;
            } else {
                failure_count++;
                std::cout << "✗ " << task.label_name
                          << " - Failed: " << task.error_message << std::endl;
            }
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Overall Statistics" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total subgraphs: " << tasks.size() << std::endl;
        std::cout << "  Successful: " << success_count << std::endl;
        std::cout << "  Failed: " << failure_count << std::endl;
        std::cout << std::endl;

        // Total pipeline time
        double preprocessing_time = classify_time + idx_gen_time;  // Mapping file generation time
        double graph_build_time = build_time;  // Graph build time
        double total_pipeline_time = preprocessing_time + graph_build_time;

        std::cout << "=== Timing Breakdown ===" << std::endl;
        std::cout << "Phase 1: Data Preprocessing (Mapping File Creation)" << std::endl;
        std::cout << "  - Label classification: " << std::fixed << std::setprecision(2)
                  << classify_time << "s" << std::endl;
        std::cout << "  - .idx file generation: " << idx_gen_time << "s" << std::endl;
        std::cout << "  - Subtotal: " << preprocessing_time << "s" << std::endl;
        std::cout << std::endl;
        std::cout << "Phase 2: Graph Build (Parallel)" << std::endl;
        std::cout << "  - Vector extraction (all subgraphs): " << std::fixed << std::setprecision(2)
                  << total_extract_time << "s (sequential equivalent)" << std::endl;
        std::cout << "  - KNN graph construction (all subgraphs): " << total_knn_time << "s (sequential equivalent)" << std::endl;
        std::cout << "  - NSG graph construction (all subgraphs): " << total_nsg_time << "s (sequential equivalent)" << std::endl;
        std::cout << "  - Actual parallel build time: " << graph_build_time << "s" << std::endl;
        std::cout << std::endl;
        std::cout << "Total Pipeline Time: " << total_pipeline_time << "s" << std::endl;
        std::cout << std::endl;

        if (success_count > 0) {
            double avg_extract = total_extract_time / success_count;
            double avg_knn = total_knn_time / success_count;
            double avg_nsg = total_nsg_time / success_count;
            std::cout << "=== Average Time (per subgraph) ===" << std::endl;
            std::cout << "  Vector extraction: " << avg_extract << "s" << std::endl;
            std::cout << "  KNN graph (NN-Descent): " << avg_knn << "s" << std::endl;
            std::cout << "  NSG graph: " << avg_nsg << "s" << std::endl;
            std::cout << "  Total: " << (avg_extract + avg_knn + avg_nsg) << "s" << std::endl;
            std::cout << std::endl;

            double sequential_time = total_extract_time + total_knn_time + total_nsg_time;
            double speedup = sequential_time / build_time;
            std::cout << "=== Parallel Efficiency ===" << std::endl;
            std::cout << "  Sequential time (estimated): " << sequential_time << "s" << std::endl;
            std::cout << "  Parallel time: " << build_time << "s" << std::endl;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
            std::cout << "  Efficiency: " << std::fixed << std::setprecision(1)
                      << (speedup / num_threads * 100.0) << "%" << std::endl;
        }

        long vmhwm_end_kb = get_vmhwm_kb();
        if (vmhwm_start_kb > 0 && vmhwm_end_kb > 0) {
            std::cout << "\n=== Memory Usage (Peak RSS) ===" << std::endl;
            std::cout << "  Initial VmHWM: " << (vmhwm_start_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Final VmHWM: " << (vmhwm_end_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Peak memory usage: " << (vmhwm_end_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Note: This includes all parallel subgraph builds and mmap overhead" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;

        return (failure_count == 0) ? 0 : -1;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return -1;
    }
}
