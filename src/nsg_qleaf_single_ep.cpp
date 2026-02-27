/**
 * @file nsg_qleaf_single_ep.cpp
 * @brief NSG Q-LEAF Single Entry Point Benchmark (Global Pool + Mapping Architecture)
 *
 */

#include "nsg_qleaf_utils.h"
#include "nsg_qleaf_data_loader.h"
#include "aligned_allocator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <stdexcept>

// EFANNA NSG headers
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

// Global distance calculation counter (single thread)
uint64_t ep_distance_calculations = 0;
uint64_t search_distance_calculations = 0;

/**
 * @brief Single search_L experiment result structure
 */
struct ExperimentResult {
    int search_L;
    size_t total_queries;
    double wall_clock_time_seconds;
    double pure_algorithm_time_seconds;
    double wall_clock_qps;
    double pure_algorithm_qps;
    double overhead_percentage;
    float recall;
    uint64_t ep_distance_calculations;
    uint64_t search_distance_calculations;
    double avg_phase1_ms;
    double avg_phase2_ms;
    double avg_phase3_ms;
    double phase1_percentage;
    double phase2_percentage;
    double phase3_percentage;
    std::vector<QueryPerformance> query_perfs;
};

/**
 * @brief Parse search_L values function
 */
std::vector<int> parse_search_L_values(const std::string& search_L_str) {
    std::vector<int> search_L_values;
    std::stringstream ss(search_L_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        // Remove whitespace
        size_t start = token.find_first_not_of(" \t\r\n");
        size_t end = token.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }

        try {
            int value = std::stoi(token);
            if (value > 0) {
                search_L_values.push_back(value);
            } else {
                std::cerr << "Warning: Ignoring non-positive search_L value: " << value << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid search_L value: " << token << std::endl;
        }
    }

    return search_L_values;
}

/**
 * @brief Save all experiment results to a single JSON file
 */
void save_all_results_json(
    const std::string& output_file,
    int k,
    const std::vector<ExperimentResult>& results) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Warning: Cannot save results to " << output_file << std::endl;
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm_now = *std::localtime(&time_t_now);

    out << "{\n";
    out << "  \"timestamp\": \"" << time_t_now << "\",\n";
    out << "  \"datetime\": \"" << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << "\",\n";
    out << "  \"framework\": \"Q-LEAF_NSG\",\n";
    out << "  \"variant\": \"single_ep\",\n";
    out << "  \"parameters\": {\n";
    out << "    \"k\": " << k << ",\n";
    out << "    \"num_entry_points\": 1,\n";
    out << "    \"search_L_values\": [";
    for (size_t i = 0; i < results.size(); i++) {
        out << results[i].search_L;
        if (i < results.size() - 1) out << ", ";
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"experiments\": [\n";

    // Results for each search_L
    for (size_t exp_idx = 0; exp_idx < results.size(); exp_idx++) {
        const auto& res = results[exp_idx];

        out << "    {\n";
        out << "      \"search_L\": " << res.search_L << ",\n";
        out << "      \"results\": {\n";
        out << "        \"total_queries\": " << res.total_queries << ",\n";
        out << "        \"wall_clock_time_seconds\": " << res.wall_clock_time_seconds << ",\n";
        out << "        \"pure_algorithm_time_seconds\": " << res.pure_algorithm_time_seconds << ",\n";
        out << "        \"wall_clock_qps\": " << res.wall_clock_qps << ",\n";
        out << "        \"pure_algorithm_qps\": " << res.pure_algorithm_qps << ",\n";
        out << "        \"overhead_percentage\": " << res.overhead_percentage << ",\n";
        out << "        \"recall@" << k << "\": " << res.recall << "\n";
        out << "      },\n";
        out << "      \"efficiency_breakdown\": {\n";
        out << "        \"ep_distance_calculations\": " << res.ep_distance_calculations << ",\n";
        out << "        \"search_distance_calculations\": " << res.search_distance_calculations << ",\n";
        out << "        \"avg_ep_distance_per_query\": "
            << static_cast<double>(res.ep_distance_calculations) / res.total_queries << ",\n";
        out << "        \"avg_search_distance_per_query\": "
            << static_cast<double>(res.search_distance_calculations) / res.total_queries << "\n";
        out << "      },\n";
        out << "      \"phase_timing\": {\n";
        out << "        \"avg_phase1_ms\": " << res.avg_phase1_ms << ",\n";
        out << "        \"avg_phase2_ms\": " << res.avg_phase2_ms << ",\n";
        out << "        \"avg_phase3_ms\": " << res.avg_phase3_ms << ",\n";
        out << "        \"phase1_percentage\": " << res.phase1_percentage << ",\n";
        out << "        \"phase2_percentage\": " << res.phase2_percentage << ",\n";
        out << "        \"phase3_percentage\": " << res.phase3_percentage << "\n";
        out << "      }\n";
        out << "    }";
        if (exp_idx < results.size() - 1) out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";

    out.close();
    std::cout << "\nAll results saved to: " << output_file << std::endl;
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "========================================" << std::endl;
        std::cerr << "NSG Q-LEAF Single Entry Point Benchmark" << std::endl;
        std::cerr << "Architecture: Global Pool + Mapping" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " <base_fvecs_file> <config_file> <query_vec_file> <query_label_file> <k> <search_L_values> <ground_truth_file>"
                  << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  base_fvecs_file    : Original base.fvecs file (shared by all subgraphs)" << std::endl;
        std::cerr << "  config_file        : Subgraph config file (4-field CSV format)" << std::endl;
        std::cerr << "  query_vec_file     : Query vectors (.fvecs)" << std::endl;
        std::cerr << "  query_label_file   : Query labels (.txt)" << std::endl;
        std::cerr << "  k                  : Number of nearest neighbors" << std::endl;
        std::cerr << "  search_L_values    : Comma-separated search_L values (e.g., \"50,100,150,200\")" << std::endl;
        std::cerr << "  ground_truth_file  : Ground truth (.ivecs)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Config file format (CSV, 4 fields per line):" << std::endl;
        std::cerr << "  category,graph_file,idx_file,entry_points_file" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " base.fvecs config.csv query.fvecs query_labels.txt 10 \"50,100,150,200\" gt.ivecs" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Note: This version uses Global Pool + Mapping architecture." << std::endl;
        std::cerr << "      All subgraphs share the same base.fvecs file via mmap." << std::endl;
        std::cerr << "      Each subgraph has its own .idx file (Local-to-Global ID mapping)." << std::endl;
        std::cerr << "========================================" << std::endl;
        return -1;
    }

    std::string base_fvecs_file = argv[1];
    std::string config_file = argv[2];
    std::string query_vec_file = argv[3];
    std::string query_label_file = argv[4];
    int k = std::stoi(argv[5]);
    std::string search_L_str = argv[6];
    std::string ground_truth_file = argv[7];

    try {
        // Parse search_L values
        std::vector<int> search_L_values = parse_search_L_values(search_L_str);

        if (search_L_values.empty()) {
            throw std::runtime_error("No valid search_L values provided");
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "NSG Q-LEAF Single Entry Point Benchmark" << std::endl;
        std::cout << "Architecture: Global Pool + Mapping" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Base file: " << base_fvecs_file << std::endl;
        std::cout << "Config file: " << config_file << std::endl;
        std::cout << "k=" << k << ", search_L values: ";
        for (size_t i = 0; i < search_L_values.size(); i++) {
            std::cout << search_L_values[i];
            if (i < search_L_values.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "Number of experiments: " << search_L_values.size() << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Global Pool method: All subgraphs share a single base.fvecs via mmap
        std::cout << "=== Loading Global Base Vectors (mmap, shared memory) ===" << std::endl;
        auto global_reader = std::make_shared<MmapVectorReader>(base_fvecs_file.c_str());
        size_t global_dim = global_reader->dimension();
        std::cout << "✓ Global base vectors loaded: " << global_reader->num_vectors()
                  << " vectors, dim=" << global_dim << std::endl;
        std::cout << "  Memory-mapped file (zero-copy, page-cached)" << std::endl;

        // MADV_WILLNEED: Preload all data to memory (for in-memory performance measurement)
        std::cout << "\n=== Preloading data to memory (MADV_WILLNEED) ===" << std::endl;
        global_reader->set_access_pattern(MADV_WILLNEED);
        std::cout << "✓ Data preloaded to memory for in-memory performance measurement" << std::endl;

        std::cout << "\n=== Loading NSG Subgraphs (Global Pool + Mapping) ===" << std::endl;

        // Phase 1: Parse config file to collect category list
        std::ifstream config_in(config_file);
        if (!config_in.is_open()) {
            throw std::runtime_error("Cannot open config file: " + config_file);
        }

        std::vector<std::vector<std::string>> config_lines;
        std::unordered_map<std::string, int> category_to_id;
        std::string line;
        int category_id = 0;

        std::cout << "Parsing config file..." << std::endl;

        while (std::getline(config_in, line)) {
            if (line.empty() || line[0] == '#') continue;

            // CSV parsing: category,graph_file,idx_file,entry_points_file
            std::vector<std::string> parts;
            std::stringstream ss(line);
            std::string part;
            while (std::getline(ss, part, ',')) {
                size_t start = part.find_first_not_of(" \t\r\n");
                size_t end = part.find_last_not_of(" \t\r\n");
                if (start != std::string::npos && end != std::string::npos) {
                    parts.push_back(part.substr(start, end - start + 1));
                } else if (start != std::string::npos) {
                    parts.push_back(part.substr(start));
                } else {
                    parts.push_back("");
                }
            }

            // Validate field count (4 fields: category, graph, idx, entry_points)
            if (parts.size() != 4) {
                std::cerr << "Warning: Invalid config line (expected 4 fields): " << line << std::endl;
                std::cerr << "  Expected format: category,graph_file,idx_file,entry_points_file" << std::endl;
                std::cerr << "  Got " << parts.size() << " fields" << std::endl;
                continue;
            }

            std::ifstream test_graph(parts[1]);
            std::ifstream test_idx(parts[2]);
            std::ifstream test_ep(parts[3]);

            if (!test_graph.is_open()) {
                std::cerr << "Warning: Graph file not found: " << parts[1] << std::endl;
            }
            if (!test_idx.is_open()) {
                std::cerr << "Warning: Index file (.idx) not found: " << parts[2] << std::endl;
            }
            if (!test_ep.is_open()) {
                std::cerr << "Warning: Entry points file not found: " << parts[3] << std::endl;
            }

            // Create category -> ID mapping
            category_to_id[parts[0]] = category_id++;
            config_lines.push_back(parts);
        }
        config_in.close();

        std::cout << "✓ Found " << category_to_id.size() << " categories in config file" << std::endl;

        if (config_lines.empty()) {
            throw std::runtime_error("No valid subgraph configurations found in config file");
        }

        // Phase 2: Load subgraphs (stored in array, Global Pool method)
        std::vector<NSGSubgraph> subgraphs(category_to_id.size());

        std::cout << "\nLoading subgraphs (Global Pool mode):" << std::endl;

        for (const auto& parts : config_lines) {
            std::cout << "  - Loading: " << parts[0] << "..." << std::endl;

            auto subgraph = load_nsg_subgraph(
                parts[0],           // category
                parts[1],           // graph_file (.graph)
                parts[2],           // idx_file (.idx, Local-to-Global ID mapping)
                parts[3],           // entry_points_file
                global_reader,      // shared_ptr to global MmapVectorReader (shared!)
                global_dim          // global dimension
            );

            int id = category_to_id[parts[0]];
            subgraphs[id] = std::move(subgraph);
            std::cout << "    ✓ Loaded: " << parts[0] << " (" << subgraphs[id].num_vectors
                      << " vectors, " << subgraphs[id].entry_points.size() << " entry points)" << std::endl;
        }

        std::cout << "\n✓ All subgraphs loaded successfully" << std::endl;

        std::cout << "\n=== Loading Queries ===" << std::endl;
        auto queries = load_queries(query_vec_file, query_label_file, global_dim, category_to_id);
        std::cout << "Loaded " << queries.size() << " queries" << std::endl;

        std::cout << "\n=== Loading Ground Truth ===" << std::endl;
        size_t gt_num, gt_dim;
        auto ground_truth = load_ground_truth(ground_truth_file, gt_num, gt_dim);
        std::cout << "Loaded ground truth: " << gt_num << " queries, " << gt_dim << " neighbors" << std::endl;

        size_t max_num_eps = 0;
        for (const auto& sg : subgraphs) {
            max_num_eps = std::max(max_num_eps, sg.entry_points.size());
        }
        AlignedVector32<float> distances_buffer(max_num_eps);
        std::cout << "Allocated 32-byte aligned distances_buffer for max " << max_num_eps << " entry points ("
                  << (max_num_eps * sizeof(float) / 1024.0) << " KB)" << std::endl;

        // Warm-up
        std::cout << "\n=== Warm-up ===" << std::endl;
        efanna2e::Parameters warmup_params;
        warmup_params.Set<unsigned>("L_search", search_L_values[0]);

        int warmup_count = std::min(100, static_cast<int>(queries.size()));
        std::cout << "Warm-up queries: " << warmup_count << " (using search_L=" << search_L_values[0] << ")" << std::endl;

        uint64_t warmup_dummy_counter = 0;

        for (int i = 0; i < warmup_count; i++) {
            const auto& q = queries[i];

            // Find subgraph
            if (q.category_id < 0 || static_cast<size_t>(q.category_id) >= subgraphs.size()) {
                std::cerr << "Warning: Warm-up query " << i << " - invalid category_id: "
                          << q.category_id << std::endl;
                continue;
            }

            auto& sg = subgraphs[q.category_id];

            // Select Entry Point
            try {
                auto [best_ep, best_ep_dist_sq] = select_best_entry_point_fast(
                    q.vector.data(), sg.entry_points,
                    sg.entry_points_data.data(),
                    sg.entry_points.size(), sg.dim,
                    warmup_dummy_counter,
                    distances_buffer.data());

                // Validate Entry Point range
                if (best_ep < 0 || static_cast<size_t>(best_ep) >= sg.num_vectors) {
                    std::cerr << "Error: Warm-up query " << i << " - invalid entry point: "
                              << best_ep << " (num_vectors: " << sg.num_vectors << ")" << std::endl;
                    continue;
                }

                // Perform NSG search
                std::vector<uint32_t> result(k);
                sg.index->SetEntryPoint(static_cast<unsigned>(best_ep));
                sg.index->SearchWithOptGraph(q.vector.data(), k, warmup_params, result.data());

                if ((i + 1) % 20 == 0) {
                    std::cout << "  Warm-up progress: " << (i + 1) << " / " << warmup_count << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during warm-up query " << i << ": " << e.what() << std::endl;
                continue;
            }
        }
        std::cout << "Warm-up completed" << std::endl;

        // Run benchmark for all search_L values
        std::vector<ExperimentResult> all_results;

        for (size_t exp_num = 0; exp_num < search_L_values.size(); exp_num++) {
            int search_L = search_L_values[exp_num];

            std::cout << "\n========================================" << std::endl;
            std::cout << "Experiment " << (exp_num + 1) << "/" << search_L_values.size()
                      << ": search_L = " << search_L << std::endl;
            std::cout << "========================================" << std::endl;

            ExperimentResult exp_result;
            exp_result.search_L = search_L;
            exp_result.total_queries = queries.size();

            std::vector<uint32_t> all_results_flat(queries.size() * k);

            std::vector<const std::vector<uint32_t>*> id_mapping_ptrs(queries.size());

            std::vector<int> query_cat_ids(queries.size());

            // Phase-wise time measurement
            std::vector<double> phase1_times(queries.size());
            std::vector<double> phase2_times(queries.size());
            std::vector<double> phase3_times(queries.size());

            // For error tracking
            std::vector<size_t> error_queries;
            std::vector<std::string> error_messages;

            // Local counters (register optimization possible, removes global variable access overhead)
            uint64_t local_ep_dist_count = 0;
            uint64_t local_search_dist_count = 0;

            // Set NSG search parameters
            efanna2e::Parameters params;
            params.Set<unsigned>("L_search", search_L);

            // Start wall-clock time measurement (measuring only the pure search loop)
            uint64_t search_dist_before_loop = efanna2e::IndexNSG::distance_computation_counter;

            auto benchmark_start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < queries.size(); i++) {
                const auto& q = queries[i];

                // Phase 1: Select subgraph (O(1) array access)
                auto phase1_start = std::chrono::high_resolution_clock::now();
                int cat_id = q.category_id;
                auto phase1_end = std::chrono::high_resolution_clock::now();
                phase1_times[i] = std::chrono::duration<double, std::milli>(phase1_end - phase1_start).count();

                // Range validation
                if (cat_id < 0 || static_cast<size_t>(cat_id) >= subgraphs.size()) {
                    error_queries.push_back(i);
                    error_messages.push_back("Invalid category_id: " + std::to_string(cat_id) +
                                           " for category: " + q.category);
                    continue;
                }

                const auto& sg = subgraphs[cat_id];

                // Phase 2: Select Entry Point (SIMD optimized)
                auto phase2_start = std::chrono::high_resolution_clock::now();

                auto [best_ep, best_ep_distance_squared] = select_best_entry_point_fast(
                    q.vector.data(), sg.entry_points,
                    sg.entry_points_data.data(),
                    sg.entry_points.size(), sg.dim,
                    local_ep_dist_count,
                    distances_buffer.data());

                auto phase2_end = std::chrono::high_resolution_clock::now();
                phase2_times[i] = std::chrono::duration<double, std::milli>(phase2_end - phase2_start).count();

                // Validate Entry Point range
                if (best_ep < 0 || static_cast<size_t>(best_ep) >= sg.num_vectors) {
                    error_queries.push_back(i);
                    error_messages.push_back("Invalid entry point: " + std::to_string(best_ep));
                    continue;
                }

                // Phase 3: NSG graph traversal (write results directly to pre-allocated location)
                auto phase3_start = std::chrono::high_resolution_clock::now();

                sg.index->SetEntryPoint(static_cast<unsigned>(best_ep));
                sg.index->SearchWithOptGraph(q.vector.data(), k, params, &all_results_flat[i * k]);

                auto phase3_end = std::chrono::high_resolution_clock::now();
                phase3_times[i] = std::chrono::duration<double, std::milli>(phase3_end - phase3_start).count();

                // Store only pointer without copying ID mapping (used for Recall calculation)
                id_mapping_ptrs[i] = &sg.id_mapping;
                query_cat_ids[i] = cat_id;
            }

            auto benchmark_end = std::chrono::high_resolution_clock::now();

            // Distance calculation counter: read only once after loop
            uint64_t search_dist_after_loop = efanna2e::IndexNSG::distance_computation_counter;
            local_search_dist_count = search_dist_after_loop - search_dist_before_loop;


            // Output error messages (after benchmark)
            if (!error_queries.empty()) {
                std::cerr << "Benchmark Errors: " << error_queries.size() << " / " << queries.size() << std::endl;
                for (size_t i = 0; i < std::min(error_queries.size(), size_t(5)); i++) {
                    std::cerr << "  Query " << error_queries[i] << ": " << error_messages[i] << std::endl;
                }
                if (error_queries.size() > 5) {
                    std::cerr << "  ... and " << (error_queries.size() - 5) << " more errors" << std::endl;
                }
            }

            // Calculate time
            exp_result.wall_clock_time_seconds =
                std::chrono::duration<double>(benchmark_end - benchmark_start).count();

            // Calculate pure algorithm time (sum of phase times)
            exp_result.pure_algorithm_time_seconds = 0.0;
            for (size_t i = 0; i < queries.size(); i++) {
                exp_result.pure_algorithm_time_seconds +=
                    (phase1_times[i] + phase2_times[i] + phase3_times[i]) / 1000.0;
            }

            exp_result.wall_clock_qps = queries.size() / exp_result.wall_clock_time_seconds;
            exp_result.pure_algorithm_qps = queries.size() / exp_result.pure_algorithm_time_seconds;
            exp_result.overhead_percentage =
                ((exp_result.wall_clock_time_seconds - exp_result.pure_algorithm_time_seconds) /
                 exp_result.wall_clock_time_seconds) * 100.0;

            // Calculate Recall
            exp_result.recall = calculate_recall_flat(
                all_results_flat.data(), id_mapping_ptrs, queries.size(),
                ground_truth, gt_dim, k);

            // Calculate average time per phase
            exp_result.avg_phase1_ms = 0.0;
            exp_result.avg_phase2_ms = 0.0;
            exp_result.avg_phase3_ms = 0.0;

            for (size_t i = 0; i < queries.size(); i++) {
                exp_result.avg_phase1_ms += phase1_times[i];
                exp_result.avg_phase2_ms += phase2_times[i];
                exp_result.avg_phase3_ms += phase3_times[i];
            }

            exp_result.avg_phase1_ms /= queries.size();
            exp_result.avg_phase2_ms /= queries.size();
            exp_result.avg_phase3_ms /= queries.size();

            double total_phase_time = exp_result.avg_phase1_ms + exp_result.avg_phase2_ms + exp_result.avg_phase3_ms;
            exp_result.phase1_percentage = (exp_result.avg_phase1_ms / total_phase_time) * 100.0;
            exp_result.phase2_percentage = (exp_result.avg_phase2_ms / total_phase_time) * 100.0;
            exp_result.phase3_percentage = (exp_result.avg_phase3_ms / total_phase_time) * 100.0;

            // Save distance calculation counters
            exp_result.ep_distance_calculations = local_ep_dist_count;
            exp_result.search_distance_calculations = local_search_dist_count;

            // Save query performance info (for detailed analysis - enable if needed)
            exp_result.query_perfs.clear();

            // Output results
            std::cout << "search_L=" << search_L << " Results:" << std::endl;
            std::cout << "  Wall-clock time: " << exp_result.wall_clock_time_seconds << " seconds" << std::endl;
            std::cout << "  Pure algorithm time: " << exp_result.pure_algorithm_time_seconds << " seconds" << std::endl;
            std::cout << "  Wall-clock QPS: " << exp_result.wall_clock_qps << std::endl;
            std::cout << "  Pure algorithm QPS: " << exp_result.pure_algorithm_qps << std::endl;
            std::cout << "  Recall@" << k << ": " << exp_result.recall << std::endl;

            all_results.push_back(std::move(exp_result));
        }

        // Output results summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "Summary of All Experiments" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::setw(10) << "search_L"
                  << std::setw(15) << "Recall@" + std::to_string(k)
                  << std::setw(15) << "Wall QPS"
                  << std::setw(15) << "Pure QPS"
                  << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;

        for (const auto& res : all_results) {
            std::cout << std::setw(10) << res.search_L
                      << std::setw(15) << std::fixed << std::setprecision(4) << res.recall
                      << std::setw(15) << std::fixed << std::setprecision(2) << res.wall_clock_qps
                      << std::setw(15) << std::fixed << std::setprecision(2) << res.pure_algorithm_qps
                      << std::endl;
        }

        // Save JSON results
        std::ostringstream timestamp_stream;
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto tm_now = *std::localtime(&time_t_now);
        timestamp_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");

        std::string output_file = "nsg_qleaf_single_ep_k" + std::to_string(k) +
                                 "_multi_L_" + timestamp_stream.str() + ".json";

        save_all_results_json(output_file, k, all_results);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
