/**
 * @file compute_entry_points_nsg_batch.cpp
 * @brief Generate Entry Points using Q-leaf method from NSG subgraph batches (sequential processing)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <memory>
#include <map>
#include <set>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include "nsg_qleaf_data_loader.h"

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
 * @brief Subgraph processing task structure
 */
struct EntryPointTask {
    std::string label_name;
    std::string graph_file;
    std::string idx_file;
    std::string output_file;

    bool success = false;
    size_t num_vectors = 0;
    size_t num_entry_points = 0;
    double load_time = 0.0;
    double kmeans_time = 0.0;
    double degree_time = 0.0;
    double total_time = 0.0;
    std::string error_message;
};

/**
 * @brief Compute L2 squared distance
 */
float l2_squared_distance(const float* v1, const float* v2, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * @brief Perform K-Means clustering and find Medoids
 */
std::vector<int64_t> find_cluster_medoids(const AlignedVector32<float>& data,
                                           size_t num_vectors,
                                           size_t dim,
                                           int num_clusters,
                                           double& kmeans_time) {
    auto start = std::chrono::high_resolution_clock::now();

    // FAISS K-Means clustering (utilizing internal parallelization)
    faiss::Clustering clustering(dim, num_clusters);
    clustering.verbose = false;
    clustering.niter = 25;
    clustering.seed = 42;

    faiss::IndexFlatL2 index(dim);
    clustering.train(num_vectors, data.data(), index);

    // Add centroids to the index
    faiss::IndexFlatL2 centroid_index(dim);
    centroid_index.add(num_clusters, clustering.centroids.data());

    // Assign each vector to the nearest centroid
    std::vector<faiss::idx_t> assignments(num_vectors);
    std::vector<float> distances(num_vectors);
    centroid_index.search(num_vectors, data.data(), 1, distances.data(), assignments.data());

    // Store the vector ID with minimum distance for each cluster (Medoid)
    std::vector<int64_t> medoids(num_clusters, -1);
    std::vector<float> min_dist_in_cluster(num_clusters, std::numeric_limits<float>::max());

    for (size_t i = 0; i < num_vectors; i++) {
        int cluster_id = static_cast<int>(assignments[i]);
        float dist = distances[i];

        if (dist < min_dist_in_cluster[cluster_id]) {
            min_dist_in_cluster[cluster_id] = dist;
            medoids[cluster_id] = i;
        }
    }

    std::vector<int64_t> valid_medoids;
    for (int64_t medoid : medoids) {
        if (medoid != -1) {
            valid_medoids.push_back(medoid);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    kmeans_time = std::chrono::duration<double>(end - start).count();

    return valid_medoids;
}

/**
 * @brief Compute node degrees from NSG graph
 */
std::vector<int> compute_degrees(efanna2e::IndexNSG* nsg_index, size_t num_vectors) {
    std::vector<int> degrees(num_vectors, 0);
    const auto& final_graph = nsg_index->GetGraph();

    for (size_t i = 0; i < num_vectors && i < final_graph.size(); i++) {
        degrees[i] = final_graph[i].size();
    }

    return degrees;
}

/**
 * @brief Select node with highest degree among Medoid and its 1-hop neighbors
 */
std::vector<int64_t> select_entry_points_by_degree(
    const std::vector<int64_t>& medoids,
    efanna2e::IndexNSG* nsg_index,
    size_t num_vectors,
    double& degree_time) {

    auto start = std::chrono::high_resolution_clock::now();

    auto degrees = compute_degrees(nsg_index, num_vectors);
    const auto& final_graph = nsg_index->GetGraph();

    std::vector<int64_t> entry_points;

    for (int64_t medoid : medoids) {
        if (medoid < 0 || medoid >= static_cast<int64_t>(num_vectors)) {
            continue;
        }

        int64_t best_node = medoid;
        int max_degree = degrees[medoid];

        if (medoid < static_cast<int64_t>(final_graph.size())) {
            for (unsigned neighbor : final_graph[medoid]) {
                if (neighbor < num_vectors) {
                    int neighbor_degree = degrees[neighbor];
                    if (neighbor_degree > max_degree) {
                        max_degree = neighbor_degree;
                        best_node = neighbor;
                    }
                }
            }
        }

        entry_points.push_back(best_node);
    }

    auto end = std::chrono::high_resolution_clock::now();
    degree_time = std::chrono::duration<double>(end - start).count();

    return entry_points;
}

/**
 * @brief Save Entry Points to file
 */
void save_entry_points(const std::string& output_file,
                      const std::vector<int64_t>& entry_points,
                      int num_clusters,
                      const std::string& graph_file,
                      const std::string& base_fvecs_file,
                      const std::string& idx_file,
                      const std::vector<uint32_t>& local_to_global_mapping) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open output file: " + output_file);
    }

    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);

    out << "# NSG Entry Points (Q-leaf Algorithm)" << std::endl;
    out << "# Generated: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
    out << "# Architecture: Global Pool + Mapping" << std::endl;
    out << "# Base file: " << base_fvecs_file << std::endl;
    out << "# Index file: " << idx_file << std::endl;
    out << "# Graph file: " << graph_file << std::endl;
    out << "# Num clusters (target): " << num_clusters << std::endl;
    out << "# Num entry points (actual): " << entry_points.size() << std::endl;
    out << "# Algorithm: K-Means + Medoid + Degree Centrality" << std::endl;
    out << "#" << std::endl;
    out << "# Entry Points (Local ID | Global ID):" << std::endl;

    for (int64_t local_ep : entry_points) {
        uint32_t global_ep = local_to_global_mapping[local_ep];
        out << local_ep << " | " << global_ep << std::endl;
    }

    out.close();
}

/**
 * @brief Generate Entry Points for a single subgraph (sequential processing)
 */
void process_single_subgraph(EntryPointTask& task,
                              const std::string& base_fvecs_file,
                              int num_clusters,
                              std::shared_ptr<MmapVectorReader>& global_base_data) {
    try {
        auto task_start = std::chrono::high_resolution_clock::now();

        std::cout << "\n[Processing] " << task.label_name << std::endl;
        std::cout << "  Graph: " << task.graph_file << std::endl;
        std::cout << "  Index: " << task.idx_file << std::endl;

        // Step 1: Load vector data
        auto load_start = std::chrono::high_resolution_clock::now();

        std::vector<uint32_t> indices = load_label_indices(task.idx_file.c_str());
        size_t num_vectors = indices.size();
        size_t dim = global_base_data->dimension();

        AlignedVector32<float> data(num_vectors * dim);
        for (size_t i = 0; i < num_vectors; i++) {
            const float* vec = global_base_data->get_vector(indices[i]);
            std::memcpy(&data[i * dim], vec, dim * sizeof(float));
        }

        auto load_end = std::chrono::high_resolution_clock::now();
        task.load_time = std::chrono::duration<double>(load_end - load_start).count();
        task.num_vectors = num_vectors;

        std::cout << "  Loaded " << num_vectors << " vectors (dim=" << dim << ")" << std::endl;

        int actual_clusters = std::min(num_clusters, static_cast<int>(num_vectors));

        // Step 2: K-Means clustering and find Medoids
        auto medoids = find_cluster_medoids(data, num_vectors, dim, actual_clusters, task.kmeans_time);
        std::cout << "  K-Means completed: " << medoids.size() << " medoids found ("
                  << task.kmeans_time << "s)" << std::endl;

        // Step 3: Load NSG graph
        auto nsg_index = std::make_shared<efanna2e::IndexNSG>(dim, num_vectors, efanna2e::L2, nullptr);
        nsg_index->Load(task.graph_file.c_str());

        // Step 4: Select Entry Points based on Degree Centrality
        auto entry_points = select_entry_points_by_degree(medoids, nsg_index.get(), num_vectors, task.degree_time);
        task.num_entry_points = entry_points.size();

        std::cout << "  Degree selection completed: " << entry_points.size() << " entry points ("
                  << task.degree_time << "s)" << std::endl;

        // Step 5: Save Entry Points
        save_entry_points(task.output_file, entry_points, num_clusters,
                         task.graph_file, base_fvecs_file, task.idx_file, indices);

        auto task_end = std::chrono::high_resolution_clock::now();
        task.total_time = std::chrono::duration<double>(task_end - task_start).count();
        task.success = true;

        std::cout << "  ✓ Completed in " << task.total_time << "s" << std::endl;

    } catch (const std::exception& e) {
        task.success = false;
        task.error_message = e.what();
        std::cerr << "  ✗ Failed: " << e.what() << std::endl;
    }
}

/**
 * @brief Scan for .graph files in directory
 */
std::vector<EntryPointTask> scan_graph_files(const std::string& graph_dir,
                                              const std::string& output_dir) {
    std::vector<EntryPointTask> tasks;

    DIR* dir = opendir(graph_dir.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open graph directory: " << graph_dir << std::endl;
        return tasks;
    }

    std::cout << "Scanning for .graph files in " << graph_dir << std::endl;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);

        if (filename.size() > 10 && filename.substr(filename.size() - 10) == "_nsg.graph") {
            std::string label = filename.substr(0, filename.size() - 10);

            EntryPointTask task;
            task.label_name = label;
            task.graph_file = graph_dir + "/" + filename;
            task.idx_file = graph_dir + "/" + label + ".idx";
            task.output_file = output_dir + "/" + label + "_entry_points.txt";

            struct stat idx_stat;
            if (stat(task.idx_file.c_str(), &idx_stat) == 0) {
                tasks.push_back(task);
                std::cout << "  Found: " << label << std::endl;
            } else {
                std::cerr << "  Warning: No .idx file for " << label << std::endl;
            }
        }
    }

    closedir(dir);

    std::sort(tasks.begin(), tasks.end(),
              [](const EntryPointTask& a, const EntryPointTask& b) {
                  return a.label_name < b.label_name;
              });

    std::cout << "Found " << tasks.size() << " subgraphs to process\n" << std::endl;
    return tasks;
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    std::string base_fvecs_file;
    std::string graph_dir;
    std::string output_dir;
    int num_clusters = 60;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base-fvecs" && i + 1 < argc) {
            base_fvecs_file = argv[++i];
        } else if (arg == "--graph-dir" && i + 1 < argc) {
            graph_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--num-clusters" && i + 1 < argc) {
            num_clusters = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --base-fvecs <file>     Original base.fvecs file" << std::endl;
            std::cout << "  --graph-dir <dir>       Directory containing .graph and .idx files" << std::endl;
            std::cout << "  --output-dir <dir>      Output directory for entry points" << std::endl;
            std::cout << "  --num-clusters <K>      Number of clusters (default: 60)" << std::endl;
            std::cout << "  --help                  Show this help message" << std::endl;
            return 0;
        }
    }

    if (base_fvecs_file.empty() || graph_dir.empty() || output_dir.empty()) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " --base-fvecs <base.fvecs> --graph-dir <dir> --output-dir <dir> [--num-clusters <K>]"
                  << std::endl;
        return -1;
    }

    try {
        std::cout << "===========================================================" << std::endl;
        std::cout << "NSG Entry Point Batch Generation (Sequential Processing)" << std::endl;
        std::cout << "===========================================================" << std::endl;
        std::cout << "Base file: " << base_fvecs_file << std::endl;
        std::cout << "Graph directory: " << graph_dir << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << "Target clusters: " << num_clusters << std::endl;
        std::cout << "Processing: SEQUENTIAL (each K-Means uses all CPU cores)" << std::endl;
        std::cout << "===========================================================\n" << std::endl;

        // Initial memory measurement
        long vmhwm_start_kb = get_vmhwm_kb();
        auto pipeline_start = std::chrono::high_resolution_clock::now();

        std::cout << "Loading global base.fvecs via mmap..." << std::endl;
        auto mmap_start = std::chrono::high_resolution_clock::now();
        auto global_base_data = std::make_shared<MmapVectorReader>(base_fvecs_file.c_str());
        auto mmap_end = std::chrono::high_resolution_clock::now();
        double mmap_time = std::chrono::duration<double>(mmap_end - mmap_start).count();

        std::cout << "✓ Loaded " << global_base_data->num_vectors() << " vectors, dim="
                  << global_base_data->dimension() << " (" << mmap_time << "s)\n" << std::endl;

        mkdir(output_dir.c_str(), 0755);

        auto tasks = scan_graph_files(graph_dir, output_dir);
        if (tasks.empty()) {
            std::cerr << "Error: No .graph files found!" << std::endl;
            return -1;
        }

        std::cout << "=== Starting Sequential Processing ===" << std::endl;
        std::cout << "Total subgraphs: " << tasks.size() << "\n" << std::endl;

        for (auto& task : tasks) {
            process_single_subgraph(task, base_fvecs_file, num_clusters, global_base_data);
        }

        auto pipeline_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(pipeline_end - pipeline_start).count();

        long vmhwm_end_kb = get_vmhwm_kb();

        std::cout << "\n===========================================================" << std::endl;
        std::cout << "Summary" << std::endl;
        std::cout << "===========================================================" << std::endl;

        int success_count = 0;
        int failure_count = 0;
        double total_load_time = 0.0;
        double total_kmeans_time = 0.0;
        double total_degree_time = 0.0;
        size_t total_vectors = 0;
        size_t total_entry_points = 0;

        for (const auto& task : tasks) {
            if (task.success) {
                success_count++;
                total_load_time += task.load_time;
                total_kmeans_time += task.kmeans_time;
                total_degree_time += task.degree_time;
                total_vectors += task.num_vectors;
                total_entry_points += task.num_entry_points;
                std::cout << "✓ " << task.label_name
                          << " (" << task.num_vectors << " vectors, "
                          << task.num_entry_points << " EPs, "
                          << std::fixed << std::setprecision(2) << task.total_time << "s)" << std::endl;
            } else {
                failure_count++;
                std::cout << "✗ " << task.label_name << " - " << task.error_message << std::endl;
            }
        }

        std::cout << "\n=== Overall Statistics ===" << std::endl;
        std::cout << "Total subgraphs: " << tasks.size() << std::endl;
        std::cout << "  Successful: " << success_count << std::endl;
        std::cout << "  Failed: " << failure_count << std::endl;
        std::cout << "Total vectors processed: " << total_vectors << std::endl;
        std::cout << "Total entry points generated: " << total_entry_points << std::endl;
        std::cout << std::endl;

        std::cout << "=== Timing Breakdown ===" << std::endl;
        std::cout << "  Global mmap load: " << std::fixed << std::setprecision(2)
                  << mmap_time << "s" << std::endl;
        std::cout << "  Vector extraction (all subgraphs): " << total_load_time << "s" << std::endl;
        std::cout << "  K-Means clustering (all subgraphs): " << total_kmeans_time << "s" << std::endl;
        std::cout << "  Degree selection (all subgraphs): " << total_degree_time << "s" << std::endl;
        std::cout << "  Total pipeline time: " << total_time << "s" << std::endl;

        if (success_count > 0) {
            std::cout << "\n=== Average Time (per subgraph) ===" << std::endl;
            std::cout << "  Vector extraction: " << (total_load_time / success_count) << "s" << std::endl;
            std::cout << "  K-Means: " << (total_kmeans_time / success_count) << "s" << std::endl;
            std::cout << "  Degree selection: " << (total_degree_time / success_count) << "s" << std::endl;
            std::cout << "  Total: " << (total_time / success_count) << "s" << std::endl;
        }

        if (vmhwm_start_kb > 0 && vmhwm_end_kb > 0) {
            std::cout << "\n=== Memory Usage (Peak RSS) ===" << std::endl;
            std::cout << "  Initial VmHWM: " << (vmhwm_start_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Final VmHWM: " << (vmhwm_end_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Peak memory usage: " << (vmhwm_end_kb / 1024.0) << " MB" << std::endl;
            std::cout << "  Note: Sequential processing with FAISS internal parallelization" << std::endl;
        }

        std::cout << "===========================================================" << std::endl;

        return (failure_count == 0) ? 0 : -1;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return -1;
    }
}
